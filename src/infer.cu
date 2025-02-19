#include "model.h"

#include <cuda_fp16.h>

#include <cfloat>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define FULL_MASK 0xffffffff

#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
			        cudaGetErrorString(err), cudaGetErrorName(err), err);                                \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

#define CUDA_CHECK2(x, msg)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "[%s] CUDA error in %s at %s:%d: %s (%s=%d)\n", msg.c_str(), __FUNCTION__, __FILE__, __LINE__, \
			        cudaGetErrorString(err), cudaGetErrorName(err), err);                                \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

static void* cuda_devicecopy(void* host, size_t size) {
	void* device = NULL;
	CUDA_CHECK(cudaMalloc(&device, size));
	CUDA_CHECK(cudaMemcpyAsync(device, host, size, cudaMemcpyHostToDevice));
	return device;
}

static void* cuda_hostcopy(void* device, size_t size, std::string debug = "") {
  void* host = NULL;
  CUDA_CHECK2(cudaMallocHost(&host, size), debug);
  CUDA_CHECK2(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost), debug);
  return host;
}

[[maybe_unused]] static void* cuda_devicealloc(size_t size) {
	void* ptr = NULL;
	CUDA_CHECK(cudaMalloc(&ptr, size));
	return ptr;
}

[[maybe_unused]] static void* cuda_hostalloc(size_t size) {
	void* ptr = NULL;
	CUDA_CHECK(cudaHostAlloc(&ptr, size, 0));
	return ptr;
}

extern "C" void* upload_cuda(void* host, size_t size) {
	return cuda_devicecopy(host, size);
}

extern "C" void* download_cuda(void* device, size_t size, std::string debug) {
  return cuda_hostcopy(device, size, debug);
}

extern "C" void register_cuda_host(void* host, size_t size) {
  CUDA_CHECK(cudaHostRegister(host, size, cudaHostRegisterDefault));
}

extern "C" void free_cuda(void* device) {
  CUDA_CHECK(cudaFree(device));
}

extern "C" void unregister_cuda_host(void* host) {
  CUDA_CHECK(cudaHostUnregister(host));
}

static int warp_size = 0;
static int max_threads_per_block = 0;
static int WARP_SIZE = 32;

extern "C" void set_cuda_device(int device) {
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device));
}

__device__ 
inline float warp_reduce_sum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);

  return val;
}

__device__ 
inline float warp_all_reduce_max(float val) {
	// Max reduction across a warp.
	// All threads will contain the max of all threads in the warp.
  for (int mask = warpSize/2; mask > 0; mask /= 2) {
    val = max(val, __shfl_xor_sync(FULL_MASK, val, mask));
  }
  return val;
}

__device__ 
inline float block_all_reduce_max(float val) {
	// Max reduction across a 1-D block implemented as double warp max reduction.
	// All threads will contain the max of all threads in the block.
	
	// Will hold results of all warps.
	// Capacity 32 since there can be at most 32 warps in a thread.
  __shared__ float shared[32];
  const int wid  = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;

  val = warp_all_reduce_max(val);

  if (blockDim.x < warpSize) return val;
  if (lane == 0) shared[wid] = val;

  __syncthreads();

  if ( wid == 0 ) {
	  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -FLT_MAX;
  }
  val = warp_all_reduce_max(val);
  if (lane == 0) shared[wid] = val;
  
  __syncthreads();
  
  return shared[0];
}

__device__ 
inline float warp_all_reduce_sum(float val) {
	// Sum reduction across a warp.
	// All threads will contain the sum of all threads in the warp.
  for (int mask = warpSize/2; mask > 0; mask /= 2) {
    val += __shfl_xor_sync(FULL_MASK, val, mask);
  }
  return val;
}

__device__ 
inline float block_all_reduce_sum(float val) {
	// Sum reduction across a 1-D block implemented as double warp sum reduction.
	// All threads will contain the sum of all threads in the block.
	
	// Will hold results of all warps.
	// Capacity 32 since there can be at most 32 warps in a thread.
  __shared__ float shared[32];
  const int wid  = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;

  val = warp_all_reduce_sum(val);

  if (blockDim.x < warpSize) return val;
  if (lane == 0) shared[wid] = val;

  __syncthreads();

  if ( wid == 0 ) {
	  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
  }
  val = warp_all_reduce_sum(val);
  if (lane == 0) shared[wid] = val;
  
  __syncthreads();
  
  return shared[0];
}

__global__
void matmul(const float* A, const float* x, int n, int d, float* out) {
	// A (d,n) @ x (n,) -> out (d,)
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= d) return;
	float sum = 0.0;
	for(int j = 0; j < n; j++){
		sum += A[n * i + j] * x[j];
	} 
	out[i] = sum;
}

__global__
void matmul(const half* A, const float* x, int n, int d, float* out) {
		// A (d,n) @ x (n,) -> out (d,)
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= d) return;
	float sum = 0.0;
	for(int j = 0; j < n; j++){
		sum += __half2float(A[n * i + j]) * x[j];
	} 
	out[i] = sum;
}



__device__
inline float matmul_row(const float* row, const float* x, int offset, int dim) {
  float sum = 0.0;
  for (int j = offset; j < dim; j += 32) {
    float v = row[j] * x[j];
    sum += v;
  }
  return warp_reduce_sum(sum);
}

__device__
inline float matmul_row(const half* row, const float* x, int offset, int dim) {
	float sum = 0.0;
	for (int j = offset; j < dim; j += 32) {
		float v = __half2float(row[j]) * x[j];
		sum += v;
	}
	return warp_reduce_sum(sum);
}

template <typename T>
__global__
inline void matmul_full_utilization(const T* A, const float* x, int n, int d, float* out){
	// A (d,n) @ x (n,) -> out (d,)
	int i = blockIdx.x;
	if (i >= d) return;
	int offset = threadIdx.x;
	float rowSum = matmul_row(&A[n * i], x, offset, n);
	if (threadIdx.x == 0) {
    	out[i] = rowSum;
	}
}

template <typename T>
__global__
inline void fused_matmul_full_utilization(const T* A, const float* x, int n, int d, float* out){
	// A (d,n) @ x (n,) -> out (d,)
	int i = blockIdx.x;
	if (i >= d) return;
	int offset = threadIdx.x;
	float rowSum = matmul_row(&A[n * i], x, offset, n);
	if (threadIdx.x == 0) {
    	out[i] += rowSum;
	}
}


__device__ inline float blocktranspose(float v, float def) {
  // Performs block-and-warp transpose operation:
  //   For a block containing K warps where lane 0 contains val_k,
  //   this function returns:
  //   - For warp 0, lane K: val_k
  //   - For all other warps and lanes: def
  int lane = threadIdx.x % warpSize;
  int warp = threadIdx.x / warpSize;
  
  // Will hold results of all warps.
  // Each lane of the warp accumulates across 1 head element at a time.
  // NOTE: Assumes warpSize is 32
  __shared__ float sm[32];
  if (lane == 0) sm[warp] = v;
  __syncthreads();
  
  return lane < blockDim.x / warpSize ? sm[lane] : def;
}

template <typename T>
__global__
void matmul_wide(const T* A, const float* x, int n, int d, float* out) {
  // A (d,n) @ x (n,) -> out (d,)
  // PRECOND: Block is 1-D and contains WPB warps.
  int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (i >= d) return;
  // Warp j computes sum for row at <blockIdx.x*WPB + j>
  // Lane 0 of each warp will hold result
  int k = threadIdx.x % warpSize; //k = offset
  float rowSum = matmul_row(&A[n * i], x, k, n);
  // Transpose values so lane k in warp 0 contains row at <blockIdx.x*WPB + k>
  // For WPB=32, this allows us to coalesce 32 float32 writes into a single 128-byte store
  rowSum = blocktranspose(rowSum, 1.0);
  if (threadIdx.x < blockDim.x / warpSize) {
    int block_start_i = blockIdx.x * blockDim.x / warpSize;
    out[block_start_i + k] = rowSum;
  }
}

template <typename T>
inline void dispatch_matmul(const T* A, const float* x, int n, int d, float* out) {// deli: n is embedding dimension, d is head dimension, kv_head_dim, or others. 
	
	/* pure naive
	int BLOCK_SIZE = 32; // arbitrary
	dim3 blocks;
	blocks.x = (n + BLOCK_SIZE - 1)/BLOCK_SIZE;
	blocks.y = d;
	dim3 tpb;
	tpb.x = BLOCK_SIZE;
	tpb.y = 1;
	matmul<<<blocks, tpb>>>(A, x, n, d, out);
	*/

	/*partial utilization
	int MAX_THREADS_PER_BLOCK = 1024;
	matmul<<<(n + MAX_THREADS_PER_BLOCK -1)/MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK>>>(A, x, n, d, out); 
	*/

	/*partial utilization2 naive matmul: thread per row
	int MAX_THREADS_PER_BLOCK = 1024;
	int BLOCK_SIZE = WARP_SIZE;
	//needs to used "d" rather than "n", in case d > n (since the called function uses "d" as threshold) 
	matmul_full_utilization<<<(d + MAX_THREADS_PER_BLOCK -1)/MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK>>>(A, x, n, d, out); 
	*/
	/*
	int BLOCK_SIZE = WARP_SIZE;
	matmul_full_utilization<<<d, BLOCK_SIZE>>>(A, x, n, d, out);
	*/
	int BLOCK_SIZE = WARP_SIZE;
	matmul_wide<<<d, BLOCK_SIZE>>>(A, x, n, d, out);
	
	
}

template <typename T>
inline void dispatch_fused_matmul(const T* A, const float* x, int n, int d, float* out) {// deli: n is embedding dimension, d is head dimension, kv_head_dim, or others. 
	int BLOCK_SIZE = WARP_SIZE;
	fused_matmul_full_utilization<<<d, BLOCK_SIZE>>>(A, x, n, d, out);
}



template <typename T>
__global__
void fused_qkv_matmul_clip(
	const T* wq,      // (q_dim, dim)
	const T* wk,      // (kv_dim, dim)
	const T* wv,      // (kv_dim, dim)
	const float* x,   // (dim,)
	int dim,          // input dimension
	int q_dim,        // n_heads * head_dim
	int kv_dim,       // n_kv_heads * head_dim
	float qkv_clip,   // clipping value
	float* q_out,     // (q_dim,)
	float* k_out,     // (kv_dim,)
	float* v_out      // (kv_dim,)
) {
	// Each warp handles one row of either Q, K, or V output
	int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	int total_rows = q_dim + 2 * kv_dim;
	if (warp_id >= total_rows) return;
	
	// Determine which matrix (Q, K, or V) we're computing
	const T* w;
	float* out;
	if (warp_id < q_dim) {
		// Computing Q
		w = wq + warp_id * dim;
		out = q_out + warp_id;
	} else if (warp_id < q_dim + kv_dim) {
		// Computing K
		w = wk + (warp_id - q_dim) * dim;
		out = k_out + (warp_id - q_dim);
	} else {
		// Computing V
		w = wv + (warp_id - q_dim - kv_dim) * dim;
		out = v_out + (warp_id - q_dim - kv_dim);
	}

	// Compute matrix multiplication for this row
	// Since block is 1-dimensional, thread ID is same as threadIdx.x,
	// and warp partitions thread IDs
	int offset = threadIdx.x % warpSize;
	float row_sum = matmul_row(w, x, offset, dim);
	// Write result with clipping
	if (offset == 0) {
		row_sum = row_sum < -qkv_clip ? -qkv_clip : (row_sum > qkv_clip ? qkv_clip : row_sum);
		*out = row_sum;
	}
}

__global__
void attn(
	const float* kb,  // (max_seq_len, n_kv_heads, head_dim) 
	const float* q,   // (n_heads, head_dim)
	int head_dim, 
	int kv_len, 
	int max_seq_len, 
	int n_heads, 
  int n_kv_heads,
	float* out        // (n_heads, kv_len)
) {
	int group = blockIdx.y;
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int h = blockIdx.y * blockDim.y + threadIdx.y;
	if (t >= kv_len || h >= n_heads) return;
	
	const float* query = q + h * head_dim;
	const float* key = kb + n_kv_heads * head_dim * t + head_dim * group;
	float score = 0.0;
	for (int i = 0; i < head_dim; i++) {
		score += query[i] * key[i];
	}
  out[h * max_seq_len + t] = score / sqrtf((float)head_dim);
}

__global__
void attn_softmax(
	const float* att, 
	int seq_len, 
	int max_seq_len, 
	int n_heads, 
	float* out
) {
	int offset = threadIdx.x;
	int h = blockIdx.x;
	int block_size = blockDim.x;
	if (h >= n_heads) return;
	
	const float* atth = att + max_seq_len * h;
	float* outh = out + max_seq_len * h;
	
	float score_max = -FLT_MAX;
	for (int t = offset; t < seq_len; t += block_size) {
		if (atth[t] > score_max) {
			score_max = atth[t];
		}
	}
	score_max = block_all_reduce_max(score_max);
	float score_sum = 0.0f;
	for (int t = offset; t < seq_len; t += block_size) {
		outh[t] = expf(atth[t] - score_max);
		score_sum += outh[t];
	}
	score_sum = block_all_reduce_sum(score_sum);
	for (int t = offset; t < seq_len; t += block_size) {
		outh[t] /= score_sum;
	}
}

/*
//naive att_mix
__global__
void att_mix(
	const float* vb,  // (max_seq_len, n_kv_heads, head_dim) 
	const float* att, // (n_heads, kv_len)
	int head_dim, 
	int n_heads, 
	int n_kv_heads,
	int seq_len, 
	int max_seq_len, 
	float* out // (n_heads, head_dim)
) {
	// PRECOND: blocks are 1-D and blockDim.x == warpSize
	int h = blockIdx.x;
	int group_size = n_heads / n_kv_heads;
	int g = h / group_size;
	int i = blockIdx.y;
	int offset = threadIdx.x;
  int kv_stride = n_kv_heads * head_dim;
	
	const float* atth = att + max_seq_len * h;
	const float* vh = vb + head_dim * g;
	float* outh = out + head_dim * h;
	
	float sum = 0.0;
	for (int t = offset; t < seq_len; t += warpSize) {
		sum += vh[kv_stride * t + i] * atth[t];
	}
	sum = warp_reduce_sum(sum);
	if (offset == 0) outh[i] = sum;
}
*/
__global__
void rmsnorm(const float* x, const float* weight, int size, float eps, float* out) {
	// PRECOND: only one 1-D block is launched
	float rms = 0.0;
	int offset = threadIdx.x;
	for (int i = offset; i < size; i += blockDim.x) {
		rms += x[i] * x[i];
	}
	rms = block_all_reduce_sum(rms);
	rms = sqrtf(rms / size + eps);
	float scale = 1.0 / rms;
	for (int i = offset; i < size; i += blockDim.x) {
		out[i] = x[i] * scale * weight[i];
	}
}

__global__
void rope(
	const float* x, int d, int head_dim, int pos, float theta, int rotary_dim, float* out
) {
	// PRECOND: grid and blocks are 1-D
	int i = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
	if (i >= d) return;
	
	int j_head = i % head_dim;
	float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
	float val = pos * freq;
	float fcr = cosf(val);
	float fci = sinf(val);
	
	float v0 = x[i];
	float v1 = x[i + 1];
	out[i] = v0 * fcr - v1 * fci;
	out[i + 1] = v0 * fci + v1 * fcr;
}

/*
//better att_mix
__global__
void att_mix(
  const float* vb,  // (max_seq_len, n_kv_heads, head_dim) 
  const float* att, // (n_heads, kv_len)
  int head_dim, 
  int n_heads, 
  int n_kv_heads,
  int seq_len, 
  int max_seq_len, 
  float* out // (n_heads, head_dim)
) {
  // PRECOND: blocks are 2-D (warp_size, t_stride)
  int h = blockIdx.x;
  int group_size = n_heads / n_kv_heads;
  int g = h / group_size;
  int kv_stride = n_kv_heads * head_dim;
  
  const float* atth = att + max_seq_len * h;
  const float* vh = vb + head_dim * g;
  float* outh = out + head_dim * h;
  
  int warp_id = threadIdx.y;
  int t_stride = blockDim.y;
  
  // Capacity 32 since there can be at most 32 warps in a block.
  __shared__ float shared[32];
  
  for (int i = threadIdx.x; i < head_dim; i += warpSize) {
    if (warp_id == 0) {
      shared[threadIdx.x] = 0;
    }
    __syncthreads();
    float sum = 0.0;
    for (int t = warp_id; t < seq_len; t += t_stride) {
      sum += vh[kv_stride * t + i] * atth[t];	
    }
    atomicAdd(&shared[threadIdx.x], sum);
    __syncthreads();
    if (warp_id == 0) {
      outh[i] = shared[threadIdx.x];
      shared[threadIdx.x] = 0;
    }
  }
}
*/

__global__
void att_mix(
  const half* vb,  // (max_seq_len, n_kv_heads, head_dim) 
  const float* att, // (n_heads, kv_len)
  int head_dim, 
  int n_heads, 
  int n_kv_heads,
  int seq_len, 
  int max_seq_len, 
  float* out // (n_heads, head_dim)
) {
  // PRECOND: blocks are 2-D (warp_size, t_stride)
  int h = blockIdx.x;
  int group_size = n_heads / n_kv_heads;
  int g = h / group_size;
  int kv_stride = n_kv_heads * head_dim;
  
  const float* atth = att + max_seq_len * h;
  const half* vh = vb + head_dim * g;
  float* outh = out + head_dim * h;
  
  int warp_id = threadIdx.y;
  int t_stride = blockDim.y;
  
  // Each lane of the warp accumulates across 2 head elements at a time.
  // NOTE: Assumes warpSize is 32
  __shared__ float shared0[32]; // shared0[i] == chunk[2*i]
  __shared__ float shared1[32]; // shared1[i] == chunk[2*i+1]
  
  for (int i = 2*threadIdx.x; i < head_dim; i += 2*warpSize) {
    if (warp_id == 0) {
      shared0[threadIdx.x] = 0;
      shared1[threadIdx.x] = 0;
    }
    __syncthreads();
    float2 sum01 = make_float2(0.0, 0.0);
    constexpr int UNROLL = 16;
    half2 v01_0; float att_0; 
    half2 v01_1; float att_1; 
    half2 v01_2; float att_2; 
    half2 v01_3; float att_3;
    half2 v01_4; float att_4;
    half2 v01_5; float att_5;
    half2 v01_6; float att_6;
    half2 v01_7; float att_7;
    half2 v01_8; float att_8; 
    half2 v01_9; float att_9; 
    half2 v01_10; float att_10; 
    half2 v01_11; float att_11;
    half2 v01_12; float att_12;
    half2 v01_13; float att_13;
    half2 v01_14; float att_14;
    half2 v01_15; float att_15;
    int t = warp_id;
    for (int ctr = 0; ctr < seq_len / t_stride; t += t_stride, ctr++) {
      int ctr_mod = ctr % UNROLL;
      if (ctr_mod == 0) {
        // prefetch every UNROLL iterations
        #define PREFETCH(j) \
          v01_##j = *((half2*)&vh[kv_stride * (t + j*t_stride) + i]); \
          att_##j = atth[t + j*t_stride];
        PREFETCH(0)
        PREFETCH(1)
        PREFETCH(2)
        PREFETCH(3)
        PREFETCH(4)
        PREFETCH(5)
        PREFETCH(6)
        PREFETCH(7)
        PREFETCH(8)
        PREFETCH(9)
        PREFETCH(10)
        PREFETCH(11)
        PREFETCH(12)
        PREFETCH(13)
        PREFETCH(14)
        PREFETCH(15)
        #undef PREFETCH
      }
      // pull one value out of prefetch batch
      float2 v01;
      float att_t;
      switch (ctr_mod) {
        #define CASE(j) \
          case j: v01 = __half22float2(v01_##j); att_t = att_##j; break;
        CASE(0)
        CASE(1)
        CASE(2)
        CASE(3)
        CASE(4)
        CASE(5)
        CASE(6)
        CASE(7)
        CASE(8)
        CASE(9)
        CASE(10)
        CASE(11)
        CASE(12)
        CASE(13)
        CASE(14)
        CASE(15)
        #undef CASE
      }
      // Sadly CUDA does not have float2 SIMD ops
      sum01.x += v01.x * att_t;
      sum01.y += v01.y * att_t;
    }
    for (; t < seq_len; t += t_stride) {
      float2 v01 = __half22float2(*((half2*)&vh[kv_stride * t + i]));
      float att_t = atth[t];
      // Sadly CUDA does not have float2 SIMD ops
      sum01.x += v01.x * att_t;
      sum01.y += v01.y * att_t;
    }
    atomicAdd(&shared0[threadIdx.x], sum01.x);
    atomicAdd(&shared1[threadIdx.x], sum01.y);
    __syncthreads();
    if (warp_id == 0) {
      float even = shared0[threadIdx.x];
      float odd = shared1[threadIdx.x];
      *((float2*)&outh[i]) = make_float2(even, odd);
      shared0[threadIdx.x] = 0;
      shared1[threadIdx.x] = 0;
    }
  }
}

__global__
void add_residuals(
	const float* x, const float* y, int d, float* out
) {
	// PRECOND: grid and blocks are 1-D
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= d) return;
	
	out[i] = x[i] + y[i];
}

__global__
void clip(
	const float* x, float v, int d, float* out
) {
	// PRECOND: grid and blocks are 1-D
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= d) return;
	
	out[i] = x[i] < -v ? -v : (x[i] > v ? v : x[i]);
}

__global__
void glu_silu(
	const float* x, const float* weight, int d, float* out
) {
	// PRECOND: grid and blocks are 1-D
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= d) return;
	
	out[i] = weight[i] * x[i] / (1.0f + expf(-x[i]));
}

__global__
void glu_gelu(
	const float* x, const float* weight, int d, float* out
) {
	// PRECOND: grid and blocks are 1-D
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= d) return;
	
	float v = x[i];
	out[i] =
		weight[i] * 
		0.5f * v * (1.0f + tanhf(0.797885f * (v + 0.044715f * v * v * v)));
}

// TODO: consolidate copy_embedding and copy_kv_entry into 1 memcpy kernel
__global__
void copy_embedding(
	const float* token_embedding_table, int dim, int token, float* out
) {
	// PRECOND: grid and blocks are 1-D
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= dim) return;
	
	const float* v = token_embedding_table + dim * token;
	out[i] = v[i];
}

__global__
void copy_embedding(
	const half* token_embedding_table, int dim, int token, float* out
) {
	// PRECOND: grid and blocks are 1-D
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= dim) return;
	
	const half* v = token_embedding_table + dim * token;
	out[i] = __half2float(v[i]);
}

__global__
void copy_kv_entry(
	const float* in, int kv_pos, int kv_dim, float* kb
) {
	// PRECOND: grid and blocks are 1-D
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= kv_dim) return;
	
	kb[kv_pos * kv_dim + i] = in[i];
}

template <typename T>
void Block::_block_cuda(
	InferenceState& s, int pos, int kv_pos, int kv_len
) const {
	const Config& c = *_config;
	
	// attention pre-norm
	switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
	    rmsnorm<<<1, max_threads_per_block>>>(
				s.x(), rms_att_weight(), c.dim, c.norm_eps, s.xb()
			);
      break;
    }
  }
  
  int q_dim = c.n_heads * c.head_dim;
  int kv_dim = c.n_kv_heads * c.head_dim;

  // qkv matmuls for this position
  dispatch_matmul<T>(wq<T>(), s.xb(), c.dim, q_dim, s.q());
  dispatch_matmul<T>(wk<T>(), s.xb(), c.dim, kv_dim, s.k());
  dispatch_matmul<T>(wv<T>(), s.xb(), c.dim, kv_dim, s.v());
  int total_rows = q_dim + 2 * kv_dim;  // Total rows across Q, K, V

  
  // some models require clipping qkv values
  clip<<<
	  (q_dim + max_threads_per_block - 1)/max_threads_per_block, 
	  max_threads_per_block
  >>>(s.q(), c.qkv_clip, q_dim, s.q());
  clip<<<
	  (kv_dim + max_threads_per_block - 1)/max_threads_per_block, 
	  max_threads_per_block
  >>>(s.k(), c.qkv_clip, kv_dim, s.k());
  clip<<<
	  (kv_dim + max_threads_per_block - 1)/max_threads_per_block,
	  max_threads_per_block
  >>>(s.v(), c.qkv_clip, kv_dim, s.v());
  
  // RoPE relative positional encoding: complex-valued rotate q and k in each head
  rope<<<
	  (q_dim + max_threads_per_block - 1)/max_threads_per_block,
	  max_threads_per_block
  >>>(
		s.q(), q_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim, s.q()
	);
	rope<<<
	  (kv_dim + max_threads_per_block - 1)/max_threads_per_block,
	  max_threads_per_block
  >>>(
		s.k(), kv_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim, s.k()
	);
	
	// key and value point to the kv cache
	float* kb = key_cache();
	float* vb = value_cache();
	half* kb = (half*)key_cache();
  	half* vb = (half*)value_cache();
	copy_kv_entry<<<
		(kv_dim + max_threads_per_block - 1)/max_threads_per_block, 
		max_threads_per_block
	>>>(
		s.k(), kv_pos, kv_dim, kb
	);
	copy_kv_entry<<<
		(kv_dim + max_threads_per_block - 1)/max_threads_per_block, 
		max_threads_per_block
	>>>(
		s.v(), kv_pos, kv_dim, vb
	);
	
	// multihead attention: dot products and softmax
	{
		dim3 tpb;
		tpb.x = warp_size;
		tpb.y = c.n_heads / c.n_kv_heads;
		dim3 blocks;
		blocks.x = (kv_len + tpb.x - 1) / tpb.x;
		blocks.y = (c.n_heads + tpb.y - 1) / tpb.y;
		attn<<<blocks, tpb>>>(
			kb, s.q(), c.head_dim, kv_len, c.max_seq_len, c.n_heads, c.n_kv_heads, s.att()
		);
		attn_softmax<<<c.n_heads, warp_size>>>(
			s.att(), kv_len, c.max_seq_len, c.n_heads, s.att()
		);
	}
	/*
  // multihead attention: mix values with attention scores
	{
		dim3 tpb;
		tpb.x = warp_size;
		dim3 blocks;
		blocks.x = c.n_heads;
		blocks.y = c.head_dim;
		att_mix<<<blocks, tpb>>>(
			vb, s.att(),
			c.head_dim, c.n_heads, c.n_kv_heads, 
			kv_len, c.max_seq_len, s.xb2()
		);
	}
	*/
	{
		dim3 tpb;
		tpb.x = warp_size;
		tpb.y = min(kv_len, max_threads_per_block / warp_size);
		dim3 blocks;
		blocks.x = c.n_heads;
		att_mix<<<blocks, tpb>>>(
		vb, s.att(),
		c.head_dim, c.n_heads, c.n_kv_heads, 
		kv_len, c.max_seq_len, s.xb2());
	}
	// final matmul projection via wo, using `hb` as temp storage
	dispatch_matmul<T>(wo<T>(), s.xb2(), q_dim, c.dim, s.hb());
	//dispatch_fused_matmul<T>(wo<T>(), s.xb2(), q_dim, c.dim, s.x());
	
	// attn residual back into x
	add_residuals<<<
		(c.dim + max_threads_per_block - 1)/max_threads_per_block, 
		max_threads_per_block
	>>>(
		s.x(), s.hb(), c.dim, s.x()
	);
	
	
	// ffn pre-norm
	switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
	    rmsnorm<<<1, max_threads_per_block>>>(
				s.x(), rms_ffn_weight(), c.dim, c.norm_eps, s.xb()
			);
      break;
    }
  }
	
	// mix self.w2(F.silu(self.w1(x)) * self.w3(x))
  // Note this is a feedforward with a GLU, not a simple MLP.
  dispatch_matmul<T>(w1<T>(), s.xb(), c.dim, c.hidden_dim, s.hb());
  dispatch_matmul<T>(w3<T>(), s.xb(), c.dim, c.hidden_dim, s.hb2());
  switch (c.act) {
	  case ActivationType::GELU: {
		  glu_gelu<<<
			  (c.hidden_dim + max_threads_per_block - 1)/max_threads_per_block, 
			  max_threads_per_block
		  >>>(
				s.hb(), s.hb2(), c.hidden_dim, s.hb()
			);
		  break;
	  }
	  case ActivationType::SILU: {
		  glu_silu<<<
			  (c.hidden_dim + max_threads_per_block - 1)/max_threads_per_block, 
			  max_threads_per_block
		  >>>(
				s.hb(), s.hb2(), c.hidden_dim, s.hb()
			);
		  break;
	  }
  }
    //dispatch_fused_matmul<T>(w2<T>(), s.hb(), c.hidden_dim, c.dim, s.x());
  	dispatch_matmul<T>(w2<T>(), s.hb(), c.hidden_dim, c.dim, s.xb2());
  
	// ffn residual back into x
	add_residuals<<<
		(c.dim + max_threads_per_block - 1)/max_threads_per_block,
		max_threads_per_block
	>>>(
		s.x(), s.xb2(), c.dim, s.x()
	);
	
}

void mha_cuda(
  float* xout,  // (n_heads, head_dim)
  float* att,   // (n_heads, max_seq_len)
  float* kb,    // (max_seq_len, n_kv_heads, head_dim)
  float* vb,    // (max_seq_len, n_kv_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
) {
  int warp_size = 32;
  // all cuda uploads leak forever...
  register_cuda_host(xout, n_heads * head_dim * sizeof(float));
  register_cuda_host(att, n_heads * max_seq_len * sizeof(float));
  kb = static_cast<float*>(upload_cuda(kb, max_seq_len * n_kv_heads * head_dim * sizeof(float)));
  vb = static_cast<float*>(upload_cuda(vb, max_seq_len * n_kv_heads * head_dim * sizeof(float)));
  q = static_cast<float*>(upload_cuda(q, n_heads * head_dim * sizeof(float)));
  // multihead attention: dot products and softmax
	{
		dim3 tpb;
		tpb.x = warp_size;
		tpb.y = n_heads / n_kv_heads;
		dim3 blocks;
		blocks.x = (kv_len + tpb.x - 1) / tpb.x;
		blocks.y = (n_heads + tpb.y - 1) / tpb.y;
		attn<<<blocks, tpb>>>(
			kb, q, head_dim, kv_len, max_seq_len, n_heads, n_kv_heads, att
		);
		attn_softmax<<<n_heads, warp_size>>>(
			att, kv_len, max_seq_len, n_heads, att
		);
	}
  // multihead attention: mix values with attention scores
	{
		dim3 tpb;
		tpb.x = warp_size;
		dim3 blocks;
		blocks.x = n_heads;
		blocks.y = head_dim;
		att_mix<<<blocks, tpb>>>(
			vb, att,
			head_dim, n_heads, n_kv_heads, 
			kv_len, max_seq_len, xout
		);
	}
  CUDA_CHECK(cudaDeviceSynchronize()); // After this, xout contains output
	CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
  unregister_cuda_host(xout);
  unregister_cuda_host(att);
}

void matmul_cuda(float* xout, float* x, float* w, int n, int d) {
  int warp_size = 32;
  int max_threads_per_block = 1024;
  // A (d,n) @ x (n,) -> out (d,)

  // all cuda uploads leak forever...
  register_cuda_host(xout, d * sizeof(float));
  x = static_cast<float*>(upload_cuda(x, n * sizeof(float)));
  w = static_cast<float*>(upload_cuda(w, n * d * sizeof(float)));
  dispatch_matmul<float>(w, x, n, d, xout);
  CUDA_CHECK(cudaDeviceSynchronize()); // After this, xout contains output
	CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
  unregister_cuda_host(xout);
}

void ffn_cuda(
  float* xout, float* x, 
  float* w1, float* w2, float* w3, 
  int hidden_dim, int dim,
  ActivationType act
) {
  int warp_size = 32;
  int max_threads_per_block = 1024;
  // all cuda uploads leak forever...
  register_cuda_host(xout, dim * sizeof(float));
  x = static_cast<float*>(upload_cuda(x, dim * sizeof(float)));
  w1 = static_cast<float*>(upload_cuda(w1, hidden_dim * dim * sizeof(float)));
  w2 = static_cast<float*>(upload_cuda(w2, dim * hidden_dim * sizeof(float)));
  w3 = static_cast<float*>(upload_cuda(w3, hidden_dim * dim * sizeof(float)));
  float* hb = new float[hidden_dim];
  float* hb2 = new float[hidden_dim];
  hb = static_cast<float*>(upload_cuda(hb, hidden_dim * sizeof(float)));
  hb2 = static_cast<float*>(upload_cuda(hb2, hidden_dim * sizeof(float)));
  // hb, hb2 leak forever on cpu too...

  // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
  // Note this is a feedforward with a GLU, not a simple MLP.
  dispatch_matmul<float>(w1, x, dim, hidden_dim, hb);
  dispatch_matmul<float>(w3, x, dim, hidden_dim, hb2);
  switch (act) {
	  case ActivationType::GELU: {
		  glu_gelu<<<
			  (hidden_dim + max_threads_per_block - 1)/max_threads_per_block, 
			  max_threads_per_block
		  >>>(
				hb, hb2, hidden_dim, hb
			);
		  break;
	  }
	  case ActivationType::SILU: {
		  glu_silu<<<
			  (hidden_dim + max_threads_per_block - 1)/max_threads_per_block, 
			  max_threads_per_block
		  >>>(
				hb, hb2, hidden_dim, hb
			);
		  break;
	  }
  }
  
  dispatch_matmul<float>(w2, hb, hidden_dim, dim, xout);
  CUDA_CHECK(cudaDeviceSynchronize()); // After this, xout contains output
	CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
  unregister_cuda_host(xout);
}

template void Block::_block_cuda<float>(InferenceState&, int, int, int) const;
template void Block::_block_cuda<half>(InferenceState&, int, int, int) const;
template<> void Block::_block_cuda<f16_t>(InferenceState& s, int pos, int kv_pos, int kv_len) const {
  _block_cuda<half>(s, pos, kv_pos, kv_len);
}

void Model::_forward_cuda(InferenceState& s, int token, int pos, InferenceMode mode) {
	const Config& c = *config;
	
  switch (c.weight_dtype) {
    case DType::F32: {
	    copy_embedding<<<
        (c.dim + max_threads_per_block - 1)/max_threads_per_block,
        max_threads_per_block
      >>>(
        static_cast<float*>(token_embedding_table), c.dim, token, s.x()
      );
      break;
    }
    case DType::F16: {
	    copy_embedding<<<
        (c.dim + max_threads_per_block - 1)/max_threads_per_block,
        max_threads_per_block
      >>>(
        static_cast<half*>(token_embedding_table), c.dim, token, s.x()
      );
      break;
    }
    default: {
      assert(false && "unsupported weight dtype for CUDA");
    }
  }
	
	// TODO: attention sinks
	int kv_pos = pos % c.max_seq_len;
	int kv_len = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;
	
	// forward all layers in order
	for (auto b : blocks) {
		b->block(s, pos, kv_pos, kv_len);
	}

  if (mode == InferenceMode::HYDRATE_KV_CACHE) {
    // only hydrate the KV cache and don't compute output logits
	  CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
    return;
  }
	
	// final layer norm
	switch (c.norm_type) {
		case LayerNormType::RMSNorm: {
			rmsnorm<<<1, max_threads_per_block>>>(
				s.x(), rms_final_weight, c.dim, c.norm_eps, s.x()
			);
			break;
		}
	}
	
	// classifier into logits
	switch (c.weight_dtype) {
    case DType::F32: {
	    dispatch_matmul<float>(
        static_cast<float*>(wcls), s.x(), c.dim, c.vocab_size, s.logits()
      );
      break;
    }
    case DType::F16: {
	    dispatch_matmul<half>(
        static_cast<half*>(wcls), s.x(), c.dim, c.vocab_size, s.logits()
      );
      break;
    }
    default: {
      assert(false && "unsupported weight dtype for CUDA");
    }
  }
	
	CUDA_CHECK(cudaDeviceSynchronize()); // After this, s.logits contains logits of output token
	CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
}