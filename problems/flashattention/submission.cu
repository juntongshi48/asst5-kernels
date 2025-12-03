// flash_attention.cu
// Template for FlashAttention CUDA Kernel Submission

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

// Constants
#define TILE_SIZE 16

// ------------------------------------------------------------------------
// CUDA Kernel Implementation
// ------------------------------------------------------------------------

template <typename scalar_t>
__global__ void naive_flash_attention_kernel(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V, scalar_t* __restrict__ O,
    const int batch_size, const int num_heads, const int seq_len,
    const int head_dim, const float scale) {
  // 1. Calculate Thread Indices
  // We map one block to one row of the Query (one token's attention)
  int bx = blockIdx.x;

  // Map linear block index to (Batch, Head, Token)
  int q_idx = bx;
  int token_idx = q_idx % seq_len;
  int head_idx = (q_idx / seq_len) % num_heads;
  int batch_idx = q_idx / (seq_len * num_heads);

  // 2. Calculate Global Memory Offsets
  // Standard layout: (Batch, Head, Seq, Dim)
  int base_offset = (batch_idx * num_heads * seq_len * head_dim) +
                    (head_idx * seq_len * head_dim);

  // Pointers to the specific row/vector for this thread
  const scalar_t* q_vec = Q + base_offset + (token_idx * head_dim);
  scalar_t* o_vec = O + base_offset + (token_idx * head_dim);

  // 3. Online Softmax State (Keep in float32 for numerical stability)
  float m_i = -INFINITY;  // Running max
  float l_i = 0.0f;       // Running sum of exponentials

  // Temporary Output Accumulator (in float32)
  // Since we can't dynamically allocate registers, we reuse the global output
  // buffer for storage, but we must be careful to read/write it as we go.
  // For this implementation, we simply assume we can overwrite O_vec.
  // Initialize Output to 0
  for (int d = 0; d < head_dim; ++d) {
    o_vec[d] = static_cast<scalar_t>(0.0f);
  }

  // 4. Iterate over Key/Value Blocks (Tiling)
  for (int j_block = 0; j_block < seq_len; j_block += TILE_SIZE) {
    // Handle edge case for last tile
    int valid_tile_size = min(TILE_SIZE, seq_len - j_block);

    // Process Tile
    for (int j = 0; j < valid_tile_size; ++j) {
      int k_idx = j_block + j;
      const scalar_t* k_vec = K + base_offset + (k_idx * head_dim);
      const scalar_t* v_vec = V + base_offset + (k_idx * head_dim);

      // A. Compute Dot Product: Q_i . K_j
      float score = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        // Cast to float for accumulation precision
        score += static_cast<float>(q_vec[d]) * static_cast<float>(k_vec[d]);
      }
      score *= scale;

      // B. Online Softmax Updates
      float m_prev = m_i;
      m_i = fmaxf(m_i, score);

      float alpha = expf(m_prev - m_i);
      float beta = expf(score - m_i);

      l_i = (l_i * alpha) + beta;

      // C. Update Output Accumulator
      // O_new = (O_old * alpha) + (V_j * beta)
      for (int d = 0; d < head_dim; ++d) {
        float o_val = static_cast<float>(o_vec[d]);
        float v_val = static_cast<float>(v_vec[d]);

        o_val = o_val * alpha + v_val * beta;

        o_vec[d] = static_cast<scalar_t>(o_val);
      }
    }
  }

  // 5. Final Normalization
  // O_final = O_acc / l_i
  for (int d = 0; d < head_dim; ++d) {
    float o_val = static_cast<float>(o_vec[d]);
    o_vec[d] = static_cast<scalar_t>(o_val / l_i);
  }
}

// ------------------------------------------------------------------------
// Helper: Block Reduction
// ------------------------------------------------------------------------
// Sums a float value across all threads in the block.
// Assumes blockDim.x is a power of 2 (standard for head_dim like 64, 128).
__inline__ __device__ float block_reduce_sum(float val) {
  extern __shared__ float shared_data[]; 
  int tid = threadIdx.x;
  
  shared_data[tid] = val;
  __syncthreads();

  // Standard Tree Reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_data[tid] += shared_data[tid + s];
    }
    __syncthreads();
  }
  
  // Grab the result
  float res = shared_data[0];
  
  // CRITICAL FIX:
  // Wait for everyone to read 'shared_data[0]' before allowing any thread 
  // to exit and potentially overwrite shared memory in the next iteration.
  __syncthreads(); 
  
  return res;
}

// ------------------------------------------------------------------------
// Kernel
// ------------------------------------------------------------------------

template <typename scalar_t>
__global__ void flash_attention_kernel(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V, scalar_t* __restrict__ O,
    const int batch_size, const int num_heads, const int seq_len,
    const int head_dim, const float scale) {
    
  // 1. Thread Setup
  // We utilize the threads inside the block now!
  int tid = threadIdx.x;  // Represents dimension 'd'
  int bx = blockIdx.x;    // Represents the Query Token

  // Map linear block index to (Batch, Head, Token)
  int q_idx = bx;
  int token_idx = q_idx % seq_len;
  int head_idx = (q_idx / seq_len) % num_heads;
  int batch_idx = q_idx / (seq_len * num_heads);

  // 2. Offsets
  int base_offset = (batch_idx * num_heads * seq_len * head_dim) +
                    (head_idx * seq_len * head_dim);

  // Pointers: Adjusted to include thread offset (tid) immediately
  // Each thread handles ONE element of the dimension.
  const scalar_t* q_ptr = Q + base_offset + (token_idx * head_dim);
  const scalar_t* k_base = K + base_offset; // We will add time offsets in loop
  const scalar_t* v_base = V + base_offset;
  scalar_t* o_ptr = O + base_offset + (token_idx * head_dim);

  // 3. Load Q into Register (Optimization: Read once, use many times)
  // Note: q_ptr[tid] corresponds to Q[batch, head, token, d]
  float q_val = static_cast<float>(q_ptr[tid]);

  // 4. Initialize State in Registers
  float m_i = -INFINITY;  // Running max
  float l_i = 0.0f;       // Running sum
  float o_accum = 0.0f;   // <-- REGISTER accumulation of Output (Fast!)

  // 5. Loop over Tiles
  for (int j_block = 0; j_block < seq_len; j_block += TILE_SIZE) {
    int valid_tile_size = min(TILE_SIZE, seq_len - j_block);

    // Process tokens in this tile
    for (int j = 0; j < valid_tile_size; ++j) {
      int k_idx = j_block + j;
      
      // Load K and V specific to this thread's dimension 'd'
      const scalar_t* k_vec = k_base + (k_idx * head_dim);
      const scalar_t* v_vec = v_base + (k_idx * head_dim);
      
      float k_val = static_cast<float>(k_vec[tid]);
      float v_val = static_cast<float>(v_vec[tid]);

      // A. Parallel Dot Product
      // Every thread computes its part of the dot product
      float partial_score = q_val * k_val;
      
      // REDUCTION: We need the sum of partial_scores across all threads
      float score = block_reduce_sum(partial_score);
      score *= scale;

      // B. Online Softmax Updates 
      // (Every thread performs this calculation redundantly, which is fine)
      float m_prev = m_i;
      m_i = fmaxf(m_i, score);
      
      float alpha = expf(m_prev - m_i);
      float beta = expf(score - m_i);
      
      l_i = (l_i * alpha) + beta;

      // C. Update Output Accumulator (in Register)
      // No atomicAdd needed because 'o_accum' is thread-local for dimension 'd'
      o_accum = o_accum * alpha + v_val * beta;
    }
  }

  // 6. Final Write to Global Memory
  // Only write once per thread at the very end!
  o_ptr[tid] = static_cast<scalar_t>(o_accum / l_i);
}

// ------------------------------------------------------------------------
// C++ / Python Interface
// ------------------------------------------------------------------------

// Required: Main function that will be called from Python
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V) {
  // 1. Setup Output Tensor
  auto O = torch::empty_like(Q);

  // 2. Extract Dimensions
  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);
  const float scale = 1.0f / sqrtf(head_dim);

  // 3. Configure Kernel Launch Parameters
  // Grid: One block per query token (Total threads = B * H * L)
  int total_threads = batch_size * num_heads * seq_len;
  
  // CONFIG CHANGE:
  // Grid remains: One block per Query Token
  // Block changes: One thread per Head Dimension
  dim3 blocks(total_threads);
  dim3 threads(head_dim); 

  // Calculate Shared Memory size needed for the reduction
  // We need sizeof(float) * head_dim bytes
  int shared_mem_size = head_dim * sizeof(float);

  // 4. Dispatch and Launch
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      Q.scalar_type(), "flash_attention_kernel", ([&] {
        flash_attention_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(), O.data_ptr<scalar_t>(), batch_size,
            num_heads, seq_len, head_dim, scale);
      }));

  // 5. Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  // 6. Synchronize to ensure kernel completion
  // (Optional for standard PyTorch usage as generic stream syncs automatically,
  // but good for strict benchmarking boundaries)
  cudaDeviceSynchronize();

  return O;
}