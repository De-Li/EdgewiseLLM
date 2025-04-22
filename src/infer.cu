#include "model.h"

#include <cuda_fp16.h>

#include <cfloat>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
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

extern "C" void set_cuda_device(int device) {
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device));
}

#if DEBUG_MODEL
#include "fmt/format.h"
static std::map<std::string, DebugTensor> _debug_map;
std::map<std::string, DebugTensor>& debug_map_cuda() {
  return _debug_map;
}
template <typename T>
static std::vector<T> copy_debug_tensor(T* device, size_t numel) {
  T* host = (T*)cuda_hostcopy(device, numel * sizeof(T));
  std::vector<T> fv(host, host + numel);
  return fv;
}
template <typename T>
static void save_debug_tensor(const std::string& name, T* x, size_t size) {
  _debug_map[name] = DebugTensor(copy_debug_tensor<T>(x, size));
}
#endif

__device__ inline float blocktranspose(float v, float def) {
  // Performs block-and-warp transpose operation:
  //   For a block containing K warps where lane 0 contains val_k,
  //   this function returns:
  //   - For warp 0, lane K: val_k
  //   - For all other warps and lanes: def
  int lane = threadIdx.x % warpSize;
  int warp = threadIdx.x / warpSize;

  // Will hold results of all warps.
  // Capacity 32 since there can be at most 32 warps in a block.
  __shared__ float sm[32];
  if (lane == 0) sm[warp] = v;
  __syncthreads();

  return lane < blockDim.x / warpSize ? sm[lane] : def;
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
  // Capacity 32 since there can be at most 32 warps in a block.
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
  // Capacity 32 since there can be at most 32 warps in a block.
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

__device__
inline float matmul_row(const float* row, const float* x, int offset, int dim) {
  float sum = 0.0;
  for (int j = offset; j < dim; j += warpSize) {
    float v = row[j] * x[j];
    sum += v;
  }
  return warp_reduce_sum(sum);
}

__device__
inline float matmul_row(const half* row, const float* x, int offset, int dim) {
  float sum = 0.0;
  for (int j = offset; j < dim; j += warpSize) {
    float v = __half2float(row[j]) * x[j];
    sum += v;
  }
  return warp_reduce_sum(sum);
}

template <typename T>
__global__
void matmul(const T* A, const float* x, int n, int d, float* out) {
  // A (d,n) @ x (n,) -> out (d,)
  // PRECOND: Block is 1-D.
  int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (i >= d) return;
  // Since block is 1-dimensional, thread ID is same as threadIdx.x,
  // and warp partitions thread IDs
  int offset = threadIdx.x % warpSize;
  float rowSum = matmul_row(&A[n * i], x, offset, n);
  if (offset == 0) {
    out[i] = rowSum;
  }
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
  int k = threadIdx.x % warpSize;
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
__global__
void fused_matmul_add_residuals(const T* A, const float* x, int n, int d, float* out) {
  // A (d,n) @ x (n,) -> out (d,)
  // PRECOND: Block is 1-D and contains WPB warps.
  int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (i >= d) return;
  // Warp j computes sum for row at <blockIdx.x*WPB + j>
  // Lane 0 of each warp will hold result
  int k = threadIdx.x % warpSize;
  float rowSum = matmul_row(&A[n * i], x, k, n);
  // Transpose values so lane k in warp 0 contains row at <blockIdx.x*WPB + k>
  // For WPB=32, this allows us to coalesce 32 float32 writes into a single 128-byte store
  rowSum = blocktranspose(rowSum, 1.0);
  if (threadIdx.x < blockDim.x / warpSize) {
    int block_start_i = blockIdx.x * blockDim.x / warpSize;
    out[block_start_i + k] += rowSum;
  }
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
inline void dispatch_matmul(const T* A, const float* x, int n, int d, float* out) {// deli: n is embedding dimension, d is head dimension, kv_head_dim, or others. 

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
  const half* kb,  // (max_seq_len, n_kv_heads, head_dim) 
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
  const half* key = kb + n_kv_heads * head_dim * t + head_dim * group;
  float score = 0.0;
  for (int i = 0; i < head_dim; i++) {
    score += query[i] * __half2float(key[i]);
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
*/
//naive
__global__
void att_mix(
  const half* vb, // (max_seq_len, n_kv_heads, head_dim) 
  const float* att, // (n_heads, kv_len)
  int head_dim, 
  int n_heads, 
  int n_kv_heads,
  int seq_len, 
  int max_seq_len, 
  float* out // (n_heads, head_dim)
) {
  // PRECOND: blocks are 1-D and blockDim.x == WARP_SIZE
  int h = blockIdx.x;
  int group_size = n_heads / n_kv_heads;
  int g = h / group_size;
  int i = blockIdx.y;
  int offset = threadIdx.x;
  int kv_stride = n_kv_heads * head_dim;
  
  const float* atth = att + max_seq_len * h;
  const half* vh = vb + head_dim * g;
  float* outh = out + head_dim * h;
  
  float sum = 0.0;
  for (int t = offset; t < seq_len; t += WARP_SIZE) {
    sum += __half2float(vh[kv_stride * t + i]) * atth[t];
  }
  sum = warp_reduce_sum(sum);
  if (offset == 0) outh[i] = sum;
}

/*
//better att_mix
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
  // PRECOND: blocks are 1-D and `out` has been zeroed
  int h = blockIdx.x;
  int group_size = n_heads / n_kv_heads;
  int g = h / group_size;
  int kv_stride = n_kv_heads * head_dim;
  
  const float* atth = att + max_seq_len * h;
  const half* vh = vb + head_dim * g;
  float* outh = out + head_dim * h;
  
  int t_per_thread = seq_len / gridDim.y;
  int t_start = blockIdx.y * t_per_thread;
  
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float sum = 0.0;
    for (int t = t_start; t < t_start + t_per_thread; t++) {
      sum += __half2float(vh[kv_stride * t + i]) * atth[t];	
    }
    atomicAdd(&outh[i], sum);
  }
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

__device__
inline void rope(
  const float* x, int pair_idx, int head_dim, int pos, float theta, int rotary_dim, float* out
) {
  int j_head = pair_idx % head_dim;
  if (j_head < head_dim - 1) {  // Ensure we have a pair of elements
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    
    float2 v01 = *((float2*)&x[pair_idx]);
    float2 result = make_float2(
      v01.x * fcr - v01.y * fci,
      v01.x * fci + v01.y * fcr
    );
    *((float2*)&out[pair_idx]) = result;
  }
}

__device__
inline void rope(
  const float* x, int pair_idx, int head_dim, int pos, float theta, int rotary_dim, half* out
) {
  int j_head = pair_idx % head_dim;
  if (j_head < head_dim - 1) {  // Ensure we have a pair of elements
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    
    float2 v01 = *((float2*)&x[pair_idx]);
    half2 result = __floats2half2_rn(
      v01.x * fcr - v01.y * fci,
      v01.x * fci + v01.y * fcr
    );
    *((half2*)&out[pair_idx]) = result;
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

__device__
inline void rope(
  const half* x, int pair_idx, int head_dim, int pos, float theta, int rotary_dim, half* out
) {
  int j_head = pair_idx % head_dim;
  if (j_head < head_dim - 1) {  // Ensure we have a pair of elements
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    
    float2 v01 = __half22float2(*((half2*)&x[pair_idx]));
    half2 result = __floats2half2_rn(
      v01.x * fcr - v01.y * fci,
      v01.x * fci + v01.y * fcr
    );
    *((half2*)&out[pair_idx]) = result;
  }
}

__global__
void copy_kv_entry(
	const half* in, int kv_pos, int kv_dim, half* kb
) {
	// PRECOND: grid and blocks are 1-D
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= kv_dim) return;
	
	kb[kv_pos * kv_dim + i] = in[i];
}

//original clip
__global__
void clip(
	const float* x, float v, int d, float* out
) {
	// PRECOND: grid and blocks are 1-D
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= d) return;
	
	out[i] = x[i] < -v ? -v : (x[i] > v ? v : x[i]);
}

template <ActivationType A> __device__ inline float act(float x);
template<> __device__ inline float act<ActivationType::SILU>(float x) {
  return x / (1.0f + expf(-x));
}
template<> __device__ inline float act<ActivationType::GELU>(float x) {
  float x3 = x * x * x;
  return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x3)));
}

template <typename T, ActivationType A>
__global__
void fused_ffn_w1_w3_glu_act(
  const T* w1,        // (hidden_dim, dim)
  const T* w3,        // (hidden_dim, dim)
  const float* x,     // (dim,)
  int dim,           
  int hidden_dim,
  float* out         // (hidden_dim,)
) {
  // Each warp computes one row of both w1(x) and w3(x), then applies GLU
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (warp_id >= hidden_dim) return;
  
  int offset = threadIdx.x % warpSize;
  
  // Compute w1(x) and w3(x) for this row
  float sum1 = matmul_row(&w1[dim * warp_id], x, offset, dim);
  float sum3 = matmul_row(&w3[dim * warp_id], x, offset, dim);
  
  // Apply activation and multiply
  if (offset == 0) {
    out[warp_id] = act<A>(sum1) * sum3;
  }
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
void fused_rope_and_cache_update(
  const float* q,         // (n_heads * head_dim,)
  const float* k,         // (n_kv_heads * head_dim,)
  const float* v,         // (n_kv_heads * head_dim,)
  int head_dim,          
  int n_heads,
  int n_kv_heads,
  int pos,               // current position
  int kv_pos,           // position in KV cache
  float theta,          // RoPE theta parameter
  int rotary_dim,       // how many dimensions to rotate
  float* q_out,         // (n_heads * head_dim,)
  half* kb,            // (max_seq_len, n_kv_heads, head_dim)
  half* vb            // (max_seq_len, n_kv_heads, head_dim)
) {
  // Each thread handles two consecutive elements (for RoPE complex rotation)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int pair_idx = tid * 2;
  
  // Handle Q matrix RoPE
  if (pair_idx < n_heads * head_dim) {
    rope(
      q, pair_idx, head_dim, pos, 
      theta, rotary_dim, q_out
    );
  }
  
  // Handle K matrix RoPE and cache update
  if (pair_idx < n_kv_heads * head_dim) {
    half* k_out = &kb[kv_pos * (n_kv_heads * head_dim)];
    rope(
      k, pair_idx, head_dim, pos, 
      theta, rotary_dim, k_out
    );
  }
  
  // Handle V cache update (no RoPE needed)
  if (pair_idx < n_kv_heads * head_dim) {
    int cache_idx = kv_pos * (n_kv_heads * head_dim) + pair_idx;
    if (pair_idx < n_kv_heads * head_dim - 1) {
      vb[cache_idx] = __float2half(v[pair_idx]);
      vb[cache_idx + 1] = __float2half(v[pair_idx + 1]);
    }
  }
}

__global__
void rotate_sink_tokens(
  half* kb, 
  int kv_sink, 				// number of attention sinks
  int kv_dim, 				// size of each entry (all concatenated heads) in KV cache
  int head_dim,
  float theta, 				// RoPE theta parameter
  int rotary_dim			// how many dimensions to rotate
) {
  // Each thread handles two consecutive elements (for RoPE complex rotation)
  // across all attention sinks
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int pair_idx = tid * 2;
  
  if (pair_idx < kv_dim) {
    for (int r = 0; r < kv_sink; r++) {
      half* k = kb + r * kv_dim;
      rope(k, pair_idx, head_dim, 1, theta, rotary_dim, k);
    }
  }
}


template <typename T>
void Block::_block_cuda(
  InferenceState& s, int pos, int kv_sink, int kv_pos, int kv_len
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

  /*
    / fused qkv matmuls for this position
    // some models require clipping qkv values
   {
    
    int total_rows = q_dim + 2 * kv_dim;  // Total rows across Q, K, V
    fused_qkv_matmul_clip<<<total_rows, warp_size>>>(
      wq<T>(),
      wk<T>(),
      wv<T>(),
      s.xb(),
      c.dim,
      q_dim,
      kv_dim,
      c.qkv_clip,
      s.q(),
      s.k(),
      s.v()
    );
  }
  */
    // qkv matmuls for this position
  dispatch_matmul<T>(wq<T>(), s.xb(), c.dim, q_dim, s.q());
  dispatch_matmul<T>(wk<T>(), s.xb(), c.dim, kv_dim, s.k());
  dispatch_matmul<T>(wv<T>(), s.xb(), c.dim, kv_dim, s.v());

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

  // Update Q, K with RoPE relative positional encoding: 
  // complex-valued rotate q and k in each head
  // Also copy K, V to KV cache
  half* kb = (half*)key_cache();
  half* vb = (half*)value_cache();
	
  //fused rope
  {
    // Calculate number of thread blocks needed
    // We need enough threads to handle the largest of:
    // - n_heads * head_dim (for Q)
    // - n_kv_heads * head_dim (for K and V)
    int max_dim = max(c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim);
    int threads_needed = (max_dim + 1) / 2;  // Each thread handles 2 elements
    int num_blocks = (threads_needed + max_threads_per_block - 1) / max_threads_per_block;
    
    fused_rope_and_cache_update<<<num_blocks, max_threads_per_block>>>(
      s.q(),
      s.k(),
      s.v(),
      c.head_dim,
      c.n_heads,
      c.n_kv_heads,
      pos,
      kv_pos,
      c.rope_theta,
      c.rotary_dim,
      s.q(),           // Q can be updated in-place
      kb,
      vb
    );
  }
  
  /*
  if (kv_sink > 0) {
    // Sink tokens remain untouched while the rest of the KV cache is incrementally 
    // replaced in ring order, but sink i must always be positioned (max_seq_len - i)
    // away from current timestep. Hence, each forward pass, rotate sink tokens 
    // forward by 1. See https://arxiv.org/abs/2309.17453 for more.
    int threads_needed = (kv_dim + 1) / 2;  // Each thread handles 2 elements
    int num_blocks = (threads_needed + max_threads_per_block - 1) / max_threads_per_block;
    rotate_sink_tokens<<<num_blocks, max_threads_per_block>>>(
      kb, kv_sink, kv_dim, c.head_dim, c.rope_theta, c.rotary_dim
    );
  }
  */
  
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
  // multihead attention: mix values with attention scores
  {
        /* usage */
    /* usage */
    dim3 tpb;
    tpb.x = WARP_SIZE;
    dim3 blocks;
    blocks.x = c.n_heads;
    blocks.y = c.head_dim;
    att_mix<<<blocks, tpb>>>(
      vb, s.att(), 
      c.head_dim, c.n_heads, c.n_kv_heads, 
      kv_len, c.max_seq_len, s.xb2()
    );
  }

  // final matmul projection and residual back:
  // x <- wo(...) + x
  /*
  fused_matmul_add_residuals<<<c.dim/32, warp_size*32>>>(
    wo<T>(), s.xb2(), q_dim, c.dim, s.x()
  );
  */
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
  
  // add residual back: x <- w2(...) + x
  /*
  fused_matmul_add_residuals<<<c.dim/32, warp_size*32>>>(
    w2<T>(), s.hb(), c.hidden_dim, c.dim, s.x()
  );
  */
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
  f16_t* kb,    // (max_seq_len, n_kv_heads, head_dim)
  f16_t* vb,    // (max_seq_len, n_kv_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
) {
  int warp_size = 32;
  int max_threads_per_block = 1024;
  // all cuda uploads leak forever...
  register_cuda_host(xout, n_heads * head_dim * sizeof(float));
  register_cuda_host(att, n_heads * max_seq_len * sizeof(float));
  kb = static_cast<f16_t*>(upload_cuda(kb, max_seq_len * n_kv_heads * head_dim * sizeof(f16_t)));
  vb = static_cast<f16_t*>(upload_cuda(vb, max_seq_len * n_kv_heads * head_dim * sizeof(f16_t)));
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
      (half*)kb, q, head_dim, kv_len, max_seq_len, n_heads, n_kv_heads, att
    );
    attn_softmax<<<n_heads, warp_size>>>(
      att, kv_len, max_seq_len, n_heads, att
    );
  }
  // multihead attention: mix values with attention scores
  {
    dim3 tpb;
    tpb.x = warp_size;
    tpb.y = min(kv_len, max_threads_per_block / warp_size);
    dim3 blocks;
    blocks.x = n_heads;
    att_mix<<<blocks, tpb>>>(
      (half*)vb, att,
      head_dim, n_heads, n_kv_heads, 
      kv_len, max_seq_len, xout
    );
  }
  CUDA_CHECK(cudaDeviceSynchronize()); // After this, xout contains output
  CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
  unregister_cuda_host(xout);
  unregister_cuda_host(att);
}

template <typename T>
void matmul_cuda(float* xout, float* x, T* w, int n, int d) {
  int warp_size = 32;
  // A (d,n) @ x (n,) -> out (d,)

  // all cuda uploads leak forever...
  register_cuda_host(xout, d * sizeof(float));
  x = static_cast<float*>(upload_cuda(x, n * sizeof(float)));
  w = static_cast<T*>(upload_cuda(w, n * d * sizeof(T)));
  matmul<<<d, warp_size>>>(w, x, n, d, xout);
  CUDA_CHECK(cudaDeviceSynchronize()); // After this, xout contains output
  CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
  unregister_cuda_host(xout);
}

template void matmul_cuda<float>(float*, float*, float*, int, int);
template void matmul_cuda<half>(float*, float*, half*, int, int);
template<> void matmul_cuda<f16_t>(float* xout, float* x, f16_t* w, int n, int d) {
  matmul_cuda<half>(xout, x, (half*)w, n, d);
}

template <typename T>
void ffn_cuda(
  float* xout, float* x, 
  T* w1, T* w2, T* w3, 
  int hidden_dim, int dim,
  ActivationType act
) {
  int warp_size = 32;
  // all cuda uploads leak forever...
  register_cuda_host(xout, dim * sizeof(float));
  x = static_cast<float*>(upload_cuda(x, dim * sizeof(float)));
  w1 = static_cast<T*>(upload_cuda(w1, hidden_dim * dim * sizeof(T)));
  w2 = static_cast<T*>(upload_cuda(w2, dim * hidden_dim * sizeof(T)));
  w3 = static_cast<T*>(upload_cuda(w3, hidden_dim * dim * sizeof(T)));
  float* hb = new float[hidden_dim];
  float* hb2 = new float[hidden_dim];
  hb = static_cast<float*>(upload_cuda(hb, hidden_dim * sizeof(float)));
  hb2 = static_cast<float*>(upload_cuda(hb2, hidden_dim * sizeof(float)));
  // hb, hb2 leak forever on cpu too...

  // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
  // Note this is a feedforward with a GLU, not a simple MLP.
  switch (act) {
    case ActivationType::GELU: {
      fused_ffn_w1_w3_glu_act<T, ActivationType::GELU><<<
        hidden_dim, warp_size
      >>>(
        w1, w3, x, dim, hidden_dim, hb
      );
      break;
    }
    case ActivationType::SILU: {
      fused_ffn_w1_w3_glu_act<T, ActivationType::SILU><<<
        hidden_dim, warp_size
      >>>(
        w1, w3, x, dim, hidden_dim, hb
      );
      break;
    }
  }
  
  matmul<<<dim, warp_size>>>(w2, hb, hidden_dim, dim, xout);
  CUDA_CHECK(cudaDeviceSynchronize()); // After this, xout contains output
  CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
  unregister_cuda_host(xout);
}

template void ffn_cuda<float>(float*, float*, float*, float*, float*, int, int, ActivationType);
template void ffn_cuda<half>(float*, float*, half*, half*, half*, int, int, ActivationType);
template <> void ffn_cuda<f16_t>(
  float* xout, float* x, 
  f16_t* w1, f16_t* w2, f16_t* w3, 
  int hidden_dim, int dim,
  ActivationType act
) {
  ffn_cuda<half>(
    xout, x, 
    (half*)w1, (half*)w2, (half*)w3, 
    hidden_dim, dim, act
  );
}

template void Block::_block_cuda<float>(InferenceState&, int, int, int, int) const;
template void Block::_block_cuda<half>(InferenceState&, int, int, int, int) const;
template<> void Block::_block_cuda<f16_t>(InferenceState& s, int pos, int kv_sink, int kv_pos, int kv_len) const {
  _block_cuda<half>(s, pos, kv_sink, kv_pos, kv_len);
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
  
  // When decoding past the context length, keep the first few tokens in the KV cache
  // untouched as "attention sinks" while replacing the rest in ring order.
  // See StreamingLLM (https://arxiv.org/pdf/2309.17453) for more.
  int kv_sink = pos >= c.max_seq_len ? KV_SINKS : 0;
  int kv_pos = kv_sink + (pos - kv_sink) % (c.max_seq_len - kv_sink);
  int kv_len = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;
  
  // forward all layers in order
  for (auto b : blocks) {
    b->block(s, pos, kv_sink, kv_pos, kv_len);
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
      matmul_wide<<<c.vocab_size/32, warp_size*32>>>(
        static_cast<float*>(wcls), s.x(), c.dim, c.vocab_size, s.logits()
      );
      break;
    }
    case DType::F16: {
      matmul_wide<<<c.vocab_size/32, warp_size*32>>>(
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