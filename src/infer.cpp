#include "model.h"

#include <assert.h>
#include <cfloat>
#include <math.h>

#include "immintrin.h"
#include "f16cintrin.h"

static void matmul(float* xout, float* x, float* w, int n, int d) {
	// W (d,n) @ x (n,) -> xout (d,)
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < d; i++) {
		float val = 0.0f;
		for (int j = 0; j < n; j++) {
			val += w[i * n + j] * x[j];
		}
		xout[i] = val;
	}
}

// matmul supporting float16 weights via the F16C extension, which allows
// conversion into float32 values before calculations.
static void matmul(float* xout, float* x, f16_t* w, int n, int d) {
#if defined(__AVX2__) && defined(__F16C__)
	// W (d,n) @ x (n,) -> xout (d,)
  assert(n % 16 == 0);
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < d; i++) {
    // Vectorized dot product of w[i][:] and x[:] where w is a packed float16 array.
		__m256 sumlo = _mm256_setzero_ps();
    __m256 sumhi = _mm256_setzero_ps();
		for (int j = 0; j < n; j+=16) {
      // Extract the next set of 16 float16 weights from `w` and store them
      // to two separate float32 vectors of width 8 (`wveclo_ps`, `wvechi_ps`)
      __m256i wvec = _mm256_loadu_si256((__m256i*)&w[i * n + j]);
      __m128i wveclo = _mm256_extractf128_si256(wvec, 0);
      __m128i wvechi = _mm256_extractf128_si256(wvec, 1);
      __m256 wveclo_ps = _mm256_cvtph_ps(wveclo);
      __m256 wvechi_ps = _mm256_cvtph_ps(wvechi);
      // Extract the next two float32 vectors of width 8 `xveclo`, `xvechi` from `x`
      __m256 xveclo = _mm256_loadu_ps(&x[j]);
      __m256 xvechi = _mm256_loadu_ps(&x[j + 8]);
      // Compute vectorized FMAs: sumlo += wveclo * xveclo, sumhi += wvechi * xvechi
      sumlo = _mm256_fmadd_ps(wveclo_ps, xveclo, sumlo);
      sumhi = _mm256_fmadd_ps(wvechi_ps, xvechi, sumhi);
		}
    // Horizontally reduce width-8 float32 vectors sumlo, sumhi to a scalar.
    __m256 sum8 = _mm256_add_ps(sumlo, sumhi);              // sum8[0:8] = sumlo[0:8] + sumhi[0:8]
    __m128 sum4 = _mm_add_ps(                               // sum4[0:4] = sum8[0:4] + sum8[4:8]
      _mm256_extractf128_ps(sum8, 0), 
      _mm256_extractf128_ps(sum8, 1)
    );
    __m128 sum1 = _mm_dp_ps(sum4, _mm_set1_ps(1.0f), 0xf1); // sum1[0] = dot(sum4, [1,1,1,1])
		xout[i] = _mm_cvtss_f32(sum1);
	}
#else
  assert(false && "float16 not supported on this platform");
#endif
}

static void rmsnorm(float* o, float* x, float* weight, int size, float eps) {
  float rms = 0.0f;
  for (int i = 0; i < size; ++i) {
    rms += x[i] * x[i];
  }
  rms = sqrtf(rms / size + eps);
  float scale = 1.0f / rms;
  for (int i = 0; i < size; ++i) {
    o[i] = x[i] * scale * weight[i];
  }
}

[[maybe_unused]] static void layernorm(float* o, float* x, float* weight, float* bias, int size, float eps) {
  float mean = 0.0f;
  for (int i = 0; i < size; ++i) {
    mean += x[i];
  }
  mean /= size;
  float var = 0.0f;
  for (int i = 0; i < size; ++i) {
    var += (x[i] - mean) * (x[i] - mean);
  }
  var /= size;
  float scale = 1.0f / sqrtf(var + eps);
  if (bias) {
    for (int i = 0; i < size; ++i) {
      o[i] = (x[i] - mean) * scale * weight[i] + bias[i];
    }
  } else {
    for (int i = 0; i < size; ++i) {
      o[i] = (x[i] - mean) * scale * weight[i];
    }
  }
}

// Compute the softmax of an input vector `x` of length `size` and store it in `o`.
static void softmax(float* o, float* x, int size) {
  float score_max = -FLT_MAX;
  for (int i = 0; i < size; ++i) {
    if (x[i] > score_max) {
      score_max = x[i];
    }
  }
  float score_sum = 0.0f;
  for (int i = 0; i < size; ++i) {
    o[i] = expf(x[i] - score_max);
    score_sum += o[i];
  }
  for (int i = 0; i < size; ++i) {
    o[i] /= score_sum;
  }
}

inline float gelu(float x) {
	return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

inline float silu(float x) {
	return x / (1.0f + expf(-x));
}

inline float clip(float x, float v) {
	return x < -v ? -v : (x > v ? v : x);
}

// TODO annotate me
static void rope(float* vec, int d, int head_dim, int pos, float theta, int rotary_dim) {
	for (int i = 0; i < d; i += 2) {
		int j_head = i % head_dim;
		float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
		float val = pos * freq;
		float fcr = cosf(val);
		float fci = sinf(val);

		float v0 = vec[i];
		float v1 = vec[i + 1];
		vec[i] = v0 * fcr - v1 * fci;
		vec[i + 1] = v0 * fci + v1 * fcr;
	}
}

// Compute next value in a sequence for a single causal self-attention head.
void attn(
  float* xout,    // (dim,) - output vector
  float* atth,    // (kv_len,) - scratch space to hold attention scores of the sequence
  float* qh,      // (head_dim,) - query vector for this head
  float* kh,      // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
  float* vh,      // (kv_len, n_kv_heads, head_dim) - buffer containing value vectors of the sequence for all KV heads
  int head_dim,   // size of the "key-space"
  int n_kv_heads, // number of kv heads, can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
  int kv_len      // number of tokens of the sequence we will attend over
) {
  int kv_stride = n_kv_heads * head_dim; // stride per token in this kv head
  // calculate attention scores as dot products of q and k
  for (int t = 0; t < kv_len; ++t) {
    float score = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      score += qh[i] * kh[t * kv_stride + i];
    }
    score /= sqrtf(head_dim);
    atth[t] = score;
  }

  // softmax the scores to get attention weights over [0..kv_len)
  softmax(atth, atth, kv_len);

  // mix values with attention weights
  for (int i = 0; i < head_dim; ++i) {
    float vi = 0.0f;
    for (int t = 0; t < kv_len; ++t) {
      vi += atth[t] * vh[t * kv_stride + i];
    }
    xout[i] = vi;
  }
}

// Compute forward pass for a single block and update the inference state accordingly.
// PRECONDITIONS: 
// - `s.x()` contains the input to the block. Output will also go here.
// - Block KV cache is hydrated.
template <typename T>
void Block::_block_cpu(
  InferenceState& s,  // inference state
  int pos,            // index of the current token in the sequence
  int kv_sink,
  int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
  int kv_len          // number of tokens in the kv cache that we will attend over
) const {
  assert(_config);
  const Config& c = *_config;

  // attention pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm(s.xb(), s.x(), rms_att_weight(), c.dim, c.norm_eps);
      break;
    }
  }

  int q_dim = c.n_heads * c.head_dim;
  int kv_dim = c.n_kv_heads * c.head_dim;

  // qkv matmuls for this position
  matmul(s.q(), s.xb(), wq<T>(), c.dim, q_dim);
  matmul(s.k(), s.xb(), wk<T>(), c.dim, kv_dim);
  matmul(s.v(), s.xb(), wv<T>(), c.dim, kv_dim);

  // some models require clipping qkv values
  for (int i = 0; i < q_dim; ++i) {
    s.q()[i] = clip(s.q()[i], c.qkv_clip);
  }
  for (int i = 0; i < kv_dim; ++i) {
    s.k()[i] = clip(s.k()[i], c.qkv_clip);
    s.v()[i] = clip(s.v()[i], c.qkv_clip);
  }

  // RoPE relative positional encoding: complex-valued rotate q and k in each head
  rope(s.q(), q_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim);
  rope(s.k(), kv_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim);
  
  // key and value point to the kv cache
  float* kb = key_cache();
  float* vb = value_cache();
  // update kv cache
  for (int i = 0; i < kv_dim; ++i) {
    kb[kv_pos * kv_dim + i] = s.k()[i];
    vb[kv_pos * kv_dim + i] = s.v()[i];
  }

  // Multihead attention. Iterate over all heads.
  int q_per_kv_head = c.n_heads / c.n_kv_heads; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < c.n_heads; h++) {
    int kv_head_offset = (h / q_per_kv_head) * c.head_dim;
    float* kh = kb + kv_head_offset;
    float* vh = vb + kv_head_offset;
    attn(s.xb2(h), s.att(h), s.q(h), kh, vh, c.head_dim, c.n_kv_heads, kv_len);
  }

  // final matmul to get output of the attention, using `hb` as temp storage
  matmul(s.hb(), s.xb2(), wo<T>(), q_dim, c.dim);

  // residual connection back into x
  for (int i = 0; i < c.dim; ++i) {
    s.x()[i] += s.hb()[i];
  }
  
  // ffn pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm(s.xb(), s.x(), rms_ffn_weight(), c.dim, c.norm_eps);
      break;
    }
  }

  // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
  // Note this is a feedforward with a GLU, not a simple MLP.
  matmul(s.hb(), s.xb(), w1<T>(), c.dim, c.hidden_dim);
  matmul(s.hb2(), s.xb(), w3<T>(), c.dim, c.hidden_dim);
  switch (c.act) {
    case ActivationType::GELU: {
      for (int i = 0; i < c.hidden_dim; ++i) {
        s.hb()[i] = gelu(s.hb()[i]) * s.hb2()[i];
      }
      break;
    }
    case ActivationType::SILU: {
      for (int i = 0; i < c.hidden_dim; ++i) {
        s.hb()[i] = silu(s.hb()[i]) * s.hb2()[i];
      }
      break;
    }
  }

  matmul(s.xb2(), s.hb(), w2<T>(), c.hidden_dim, c.dim);
  // residual connection back into x
  for (int i = 0; i < c.dim; ++i) {
    s.x()[i] += s.xb2()[i];
  }
}

void mha_cpu(
  float* xout,  // (n_heads, head_dim)
  float* att,   // (n_heads, max_seq_len)
  float* kb,    // (max_seq_len, n_kv_heads, head_dim)
  float* vb,    // (max_seq_len, n_kv_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
) {
  // Multihead attention. Iterate over all heads.
  int q_per_kv_head = n_heads / n_kv_heads; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < n_heads; h++) {
    int kv_head_offset = (h / q_per_kv_head) * head_dim;
    float* kh = kb + kv_head_offset;
    float* vh = vb + kv_head_offset;
    attn(
      xout + head_dim * h, att + max_seq_len * h, q + head_dim * h, 
      kh, vh, head_dim, n_kv_heads, kv_len
    );
  }
}

void matmul_cpu(float* xout, float* x, float* w, int n, int d) {
  matmul(xout, x, w, n, d);
}
void matmul_cpu(float* xout, float* x, f16_t* w, int n, int d) {
  matmul(xout, x, w, n, d);
}

void ffn_cpu(
  float* xout, float* x, 
  float* w1, float* w2, float* w3, 
  int hidden_dim, int dim,
  ActivationType act
) {
  float* hb = new float[hidden_dim];
  float* hb2 = new float[hidden_dim];
  // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
  // Note this is a feedforward with a GLU, not a simple MLP.
  matmul(hb, x, w1, dim, hidden_dim);
  matmul(hb2, x, w3, dim, hidden_dim);
  switch (act) {
    case ActivationType::GELU: {
      for (int i = 0; i < hidden_dim; ++i) {
        hb[i] = gelu(hb[i]) * hb2[i];
      }
      break;
    }
    case ActivationType::SILU: {
      for (int i = 0; i < hidden_dim; ++i) {
        hb[i] = silu(hb[i]) * hb2[i];
      }
      break;
    }
  }

  matmul(xout, hb, w2, hidden_dim, dim);
  
  delete[] hb;
  delete[] hb2;
}

template void Block::_block_cpu<float>(InferenceState&, int, int, int, int) const;
template void Block::_block_cpu<f16_t>(InferenceState&, int, int, int, int) const;

void Model::_copy_embedding(InferenceState& s, int token) {
  const Config& c = *config;
  switch (c.weight_dtype) {
    case DType::F32: {
      float* emb = static_cast<float*>(token_embedding_table);
      for (int i = 0; i < c.dim; ++i) {
        s.x()[i] = emb[token * c.dim + i];
      }
      break;
    }
    case DType::F16: {
#if defined(__AVX2__) && defined(__F16C__)
      f16_t* emb = static_cast<f16_t*>(token_embedding_table);
      for (int i = 0; i < c.dim; i+=1) {
        s.x()[i] = _cvtsh_ss(emb[token * c.dim + i]);
      }
#else
      assert(false && "float16 not supported on this platform");
#endif
      break;
    }
    default: {
      assert(false && "unsupported weight dtype");
    }
  }
}

void Model::_forward_cpu(InferenceState& s, int token, int pos, InferenceMode mode) {
  const Config& c = *config;

  // copy the token embedding into `x`
  _copy_embedding(s, token);

  // TODO: attention sinks
	int kv_pos = pos % c.max_seq_len;
	int kv_len = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;
  int fillingtemp = 0;
  // forward all layers in order
  for (auto b : blocks) {
    b->block(s, pos, fillingtemp, kv_pos, kv_len);
  }

  if (mode == InferenceMode::HYDRATE_KV_CACHE) {
    // only hydrate the KV cache and don't compute output logits
    return;
  }

  // final layer norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm(s.x(), s.x(), rms_final_weight, c.dim, c.norm_eps);
      break;
    }
  }

  // classifier into logits
  switch (c.weight_dtype) {
    case DType::F32: {
      matmul(s.logits(), s.x(), static_cast<float*>(wcls), c.dim, c.vocab_size);
      break;
    }
    case DType::F16: {
      matmul(s.logits(), s.x(), static_cast<f16_t*>(wcls), c.dim, c.vocab_size);
      break;
    }
    default: {
      assert(false && "unsupported weight dtype");
    }
  }
}