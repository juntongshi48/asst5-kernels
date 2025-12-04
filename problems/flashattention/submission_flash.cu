#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

// Configuration - tuned for the given test case
// batch_size: 4; num_heads: 64; seq_len: 8192; head_dim: 128
#define BR 16      // Query rows per block (reduced for register pressure)
#define BC 16      // Key/Value cols per tile
#define BK 128     // Head dimension (fixed)
#define NUM_THREADS 128

// Warp configuration
#define WARP_SIZE 32
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)  // 4 warps

// ------------------------------------------------------------------------
// Flash Attention Kernel
// 
// True Flash Attention implementation:
// - Loads tiles of Q, K, V from HBM to shared memory (SRAM)
// - Uses online softmax algorithm for numerical stability
// - O(N) memory complexity (does not materialize full attention matrix)
// - Single pass over K, V with streaming
// ------------------------------------------------------------------------

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
    
  // Thread indices
  const int tid = threadIdx.x;          // 0 to 127, corresponds to head dimension
  const int q_tile_idx = blockIdx.x;    // Which Q tile
  const int bh_idx = blockIdx.y;        // batch * num_heads index
  
  const int batch_idx = bh_idx / num_heads;
  const int head_idx = bh_idx % num_heads;
  
  // Calculate base pointers for this (batch, head)
  const int offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
  const scalar_t* Q_base = Q + offset;
  const scalar_t* K_base = K + offset;
  const scalar_t* V_base = V + offset;
  scalar_t* O_base = O + offset;
  
  // Q tile boundaries
  const int q_start = q_tile_idx * BR;
  const int q_end = min(q_start + BR, seq_len);
  const int num_rows = q_end - q_start;
  
  // Shared memory layout
  extern __shared__ float smem[];
  float* Q_tile = smem;                                  // BR x BK
  float* K_tile = smem + BR * BK;                        // BC x BK
  float* V_tile = smem + BR * BK + BC * BK;              // BC x BK
  float* S_tile = smem + BR * BK + 2 * BC * BK;          // BR x BC
  float* warp_scratch = smem + BR * BK + 2 * BC * BK + BR * BC;  // NUM_WARPS floats
  
  // Register arrays for online softmax state
  // Each thread maintains state for all BR query rows
  // Thread tid handles output dimension tid
  float row_max[BR];   // Running maximum for softmax
  float row_sum[BR];   // Running sum of exp(score - max)
  float out_acc[BR];   // Output accumulator for dimension tid
  
  // Initialize
  #pragma unroll
  for (int r = 0; r < BR; r++) {
    row_max[r] = -INFINITY;
    row_sum[r] = 0.0f;
    out_acc[r] = 0.0f;
  }
  
  // ============ Load Q tile into shared memory ============
  // Q_tile[r][d] = Q[q_start + r][d]
  // Total: BR * BK elements, 128 threads
  for (int idx = tid; idx < BR * BK; idx += NUM_THREADS) {
    int r = idx / BK;
    int d = idx % BK;
    int global_r = q_start + r;
    Q_tile[idx] = (global_r < seq_len) ? 
                  static_cast<float>(Q_base[global_r * head_dim + d]) : 0.0f;
  }
  __syncthreads();
  
  // ============ Main loop: iterate over K/V tiles ============
  const int num_kv_tiles = (seq_len + BC - 1) / BC;
  
  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    const int kv_start = kv_tile * BC;
    const int kv_end = min(kv_start + BC, seq_len);
    const int num_cols = kv_end - kv_start;
    
    // Load K tile
    for (int idx = tid; idx < BC * BK; idx += NUM_THREADS) {
      int r = idx / BK;
      int d = idx % BK;
      int global_r = kv_start + r;
      K_tile[idx] = (global_r < seq_len) ?
                    static_cast<float>(K_base[global_r * head_dim + d]) : 0.0f;
    }
    
    // Load V tile
    for (int idx = tid; idx < BC * BK; idx += NUM_THREADS) {
      int r = idx / BK;
      int d = idx % BK;
      int global_r = kv_start + r;
      V_tile[idx] = (global_r < seq_len) ?
                    static_cast<float>(V_base[global_r * head_dim + d]) : 0.0f;
    }
    __syncthreads();
    
    // ============ Compute S = Q @ K^T ============
    // S_tile[i][j] = scale * sum_d(Q_tile[i][d] * K_tile[j][d])
    // Strategy: Each thread computes partial products for dimension tid,
    // then we reduce across all 128 threads
    
    for (int i = 0; i < num_rows; i++) {
      for (int j = 0; j < num_cols; j++) {
        // Partial dot product: Q[i, tid] * K[j, tid]
        float partial = Q_tile[i * BK + tid] * K_tile[j * BK + tid];
        
        // Warp-level reduction using shuffle
        #pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
          partial += __shfl_xor_sync(0xffffffff, partial, mask);
        }
        
        // Now each thread in warp has the warp sum
        // Lane 0 of each warp writes to scratch
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        
        if (lane_id == 0) {
          warp_scratch[warp_id] = partial;
        }
        __syncthreads();
        
        // Thread 0 does final reduction across warps
        if (tid == 0) {
          float sum = 0.0f;
          #pragma unroll
          for (int w = 0; w < NUM_WARPS; w++) {
            sum += warp_scratch[w];
          }
          S_tile[i * BC + j] = sum * scale;
        }
        __syncthreads();
      }
    }
    
    // ============ Online Softmax + Output Accumulation ============
    for (int i = 0; i < num_rows; i++) {
      // Find max of new scores for row i
      float new_max = -INFINITY;
      for (int j = 0; j < num_cols; j++) {
        new_max = fmaxf(new_max, S_tile[i * BC + j]);
      }
      
      // Compute updated running max
      float m_new = fmaxf(row_max[i], new_max);
      
      // Correction factor for previous accumulations
      float correction = expf(row_max[i] - m_new);
      
      // Rescale previous values
      row_sum[i] *= correction;
      out_acc[i] *= correction;
      
      // Add contributions from this K/V tile
      for (int j = 0; j < num_cols; j++) {
        float p = expf(S_tile[i * BC + j] - m_new);
        row_sum[i] += p;
        // Each thread handles dimension tid
        out_acc[i] += p * V_tile[j * BK + tid];
      }
      
      row_max[i] = m_new;
    }
    __syncthreads();
  }
  
  // ============ Write final output ============
  for (int i = 0; i < num_rows; i++) {
    int global_row = q_start + i;
    if (global_row < seq_len) {
      // Normalize by sum of exponentials
      float val = (row_sum[i] > 0.0f) ? (out_acc[i] / row_sum[i]) : 0.0f;
      O_base[global_row * head_dim + tid] = static_cast<scalar_t>(val);
    }
  }
}

// ------------------------------------------------------------------------
// C++ / Python Interface
// ------------------------------------------------------------------------

torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V) {
  // Create output tensor
  auto O = torch::empty_like(Q);

  // Get dimensions
  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  // Grid and block dimensions
  const int num_q_tiles = (seq_len + BR - 1) / BR;
  dim3 grid(num_q_tiles, batch_size * num_heads);
  dim3 block(NUM_THREADS);  // 128 threads = head_dim
  
  // Shared memory size calculation
  // Q_tile: BR * BK = 32 * 128 = 4096
  // K_tile: BC * BK = 32 * 128 = 4096  
  // V_tile: BC * BK = 32 * 128 = 4096
  // S_tile: BR * BC = 32 * 32 = 1024
  // warp_scratch: NUM_WARPS = 4
  size_t smem_size = (BR * BK + BC * BK + BC * BK + BR * BC + NUM_WARPS) * sizeof(float);

  // Launch kernel
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

  // Error checking
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  cudaDeviceSynchronize();

  return O;
}
