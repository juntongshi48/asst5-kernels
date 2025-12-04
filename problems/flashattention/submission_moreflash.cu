#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

// Tile sizes
#define BR 16
#define BC 16  
#define BK 128

// Thread configuration: 128 threads total
// Each thread handles one dimension of the head
#define NUM_THREADS 128
#define WARP_SIZE 32
#define NUM_WARPS 4

template <typename scalar_t>
__global__ void flash_attention_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale) {
    
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  
  const int q_tile_idx = blockIdx.x;
  const int bh_idx = blockIdx.y;
  
  const int batch_idx = bh_idx / num_heads;
  const int head_idx = bh_idx % num_heads;
  
  const long long offset = ((long long)batch_idx * num_heads + head_idx) * seq_len * head_dim;
  const scalar_t* Q_base = Q + offset;
  const scalar_t* K_base = K + offset;
  const scalar_t* V_base = V + offset;
  scalar_t* O_base = O + offset;
  
  const int q_start = q_tile_idx * BR;
  const int num_rows = min(BR, seq_len - q_start);
  
  if (num_rows <= 0) return;
  
  // Shared memory layout
  extern __shared__ float smem[];
  float* Q_smem = smem;                    // BR x BK
  float* K_smem = Q_smem + BR * BK;        // BC x BK
  float* V_smem = K_smem + BC * BK;        // BC x BK
  float* S_smem = V_smem + BC * BK;        // BR x BC (scores)
  
  // Online softmax state per row (in registers)
  float row_max[BR];
  float row_sum[BR];
  float out_acc[BR];  // Output for dimension tid
  
  #pragma unroll
  for (int i = 0; i < BR; i++) {
    row_max[i] = -INFINITY;
    row_sum[i] = 0.0f;
    out_acc[i] = 0.0f;
  }
  
  // Load Q tile (coalesced)
  for (int idx = tid; idx < BR * BK; idx += NUM_THREADS) {
    int r = idx / BK;
    int d = idx % BK;
    int global_r = q_start + r;
    Q_smem[idx] = (global_r < seq_len) ? 
                  static_cast<float>(Q_base[global_r * head_dim + d]) : 0.0f;
  }
  __syncthreads();
  
  // Process K/V tiles
  const int num_kv_tiles = (seq_len + BC - 1) / BC;
  
  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    const int kv_start = kv_tile * BC;
    const int num_cols = min(BC, seq_len - kv_start);
    
    // Load K and V (coalesced)
    for (int idx = tid; idx < BC * BK; idx += NUM_THREADS) {
      int r = idx / BK;
      int d = idx % BK;
      int global_r = kv_start + r;
      K_smem[idx] = (global_r < seq_len) ? 
                    static_cast<float>(K_base[global_r * head_dim + d]) : 0.0f;
      V_smem[idx] = (global_r < seq_len) ? 
                    static_cast<float>(V_base[global_r * head_dim + d]) : 0.0f;
    }
    __syncthreads();
    
    // ========== Compute S = Q @ K^T ==========
    // Parallelize: 128 threads compute 128 scores simultaneously
    // BR*BC = 1024 scores, need 8 iterations
    
    for (int score_offset = 0; score_offset < BR * BC; score_offset += NUM_THREADS) {
      int score_idx = score_offset + tid;
      int i = score_idx / BC;
      int j = score_idx % BC;
      
      float score = 0.0f;
      if (i < num_rows && j < num_cols) {
        // Compute dot product in registers
        #pragma unroll 16
        for (int d = 0; d < BK; d++) {
          score += Q_smem[i * BK + d] * K_smem[j * BK + d];
        }
        score *= scale;
      } else {
        score = -INFINITY;  // Masked position
      }
      
      if (score_idx < BR * BC) {
        S_smem[score_idx] = score;
      }
    }
    __syncthreads();
    
    // ========== Online Softmax + Output ==========
    // Each thread processes all rows but only its dimension tid for output
    
    for (int i = 0; i < num_rows; i++) {
      // Find max of this row's new scores
      // Thread tid reads score at position tid % BC, then we reduce
      float my_score = (tid < num_cols) ? S_smem[i * BC + tid] : -INFINITY;
      
      // Additional scores for threads that handle multiple
      float extra_score = -INFINITY;
      if (tid + 32 < num_cols) extra_score = fmaxf(extra_score, S_smem[i * BC + tid + 32]);
      if (tid + 64 < num_cols) extra_score = fmaxf(extra_score, S_smem[i * BC + tid + 64]);
      if (tid + 96 < num_cols) extra_score = fmaxf(extra_score, S_smem[i * BC + tid + 96]);
      
      float local_max = fmaxf(my_score, extra_score);
      
      // Warp reduction for max
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
      }
      
      // Cross-warp reduction
      __shared__ float warp_max[NUM_WARPS];
      if (lane_id == 0) warp_max[warp_id] = local_max;
      __syncthreads();
      
      float new_max = warp_max[0];
      #pragma unroll
      for (int w = 1; w < NUM_WARPS; w++) {
        new_max = fmaxf(new_max, warp_max[w]);
      }
      
      // Online softmax update
      float m_new = fmaxf(row_max[i], new_max);
      float correction = expf(row_max[i] - m_new);
      
      row_sum[i] *= correction;
      out_acc[i] *= correction;
      
      // Compute sum of exp and accumulate output
      float local_sum = 0.0f;
      for (int j = 0; j < num_cols; j++) {
        float p = expf(S_smem[i * BC + j] - m_new);
        local_sum += p;
        out_acc[i] += p * V_smem[j * BK + tid];
      }
      
      row_sum[i] += local_sum;
      row_max[i] = m_new;
    }
    __syncthreads();
  }
  
  // Write output
  for (int i = 0; i < num_rows; i++) {
    int global_row = q_start + i;
    if (global_row < seq_len && row_sum[i] > 0.0f) {
      O_base[global_row * head_dim + tid] = static_cast<scalar_t>(out_acc[i] / row_sum[i]);
    }
  }
}

torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V) {
  Q = Q.contiguous();
  K = K.contiguous();
  V = V.contiguous();
  
  auto O = torch::empty_like(Q);

  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  const int num_q_tiles = (seq_len + BR - 1) / BR;
  dim3 grid(num_q_tiles, batch_size * num_heads);
  dim3 block(NUM_THREADS);
  
  size_t smem_size = (BR * BK + BC * BK + BC * BK + BR * BC + NUM_WARPS) * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      Q.scalar_type(), "flash_attention_kernel", ([&] {
        flash_attention_kernel<scalar_t><<<grid, block, smem_size>>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            O.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            scale);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  return O;
}
