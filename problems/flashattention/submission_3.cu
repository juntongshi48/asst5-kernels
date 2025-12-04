// flash_attention.cu
// Template for FlashAttention CUDA Kernel Submission

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <vector>

// Constants
#define TILE_SIZE 16

// Constants for Tiling
// We process 'Bc' queries in parallel per block
// We load 'Br' keys/values into shared memory per loop step
#define Bc 8   
#define Br 32  

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

__inline__ __device__ float warp_reduce_sum(float val) {
    // 0xffffffff means all threads in the warp participate
    // We shift down by 16, 8, 4, 2, 1 to sum everything into lane 0
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void flash_attention_kernel_tiled(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V, scalar_t* __restrict__ O,
    const int batch_size, const int num_heads, const int seq_len,
    const int head_dim, const float scale) {

  // --------------------------------------------------------
  // 1. Setup Shared Memory
  // --------------------------------------------------------
  // We need space for K_tile, V_tile, and a reduction buffer for EACH query row
  extern __shared__ float smem[];
  
  // Pointers into shared memory
  // layout: K_tile[Br][head_dim], then V_tile[Br][head_dim]
  float* K_tile = smem; 
  float* V_tile = &smem[Br * head_dim]; 
  // Reduction buffer: Bc rows, each needs 1 float for the partial sum result? 
  // Actually, for block_reduce_sum, we need scratch space.
  // Let's rely on the helper to manage its own reduction shmem or pass it in.
  // For simplicity here, let's assume we pass a pointer to the helper.
  float* red_smem = &smem[2 * Br * head_dim]; 

  // --------------------------------------------------------
  // 2. Thread Indexing
  // --------------------------------------------------------
  int tx = threadIdx.x; // Dimension 0..head_dim
  int ty = threadIdx.y; // Q-Token in Tile 0..Bc

  // Identify which global query token this thread handles
  // Grid is now (Batch * Heads * (SeqLen / Bc))
  int tile_idx = blockIdx.x; 
  
  // Derive offsets (Batch, Head, Seq start)
  // Note: This math assumes SeqLen is divisible by Bc for simplicity
  int num_tiles_seq = (seq_len + Bc - 1) / Bc;
  int batch_head_idx = tile_idx / num_tiles_seq;
  int q_tile_start = (tile_idx % num_tiles_seq) * Bc;

  int q_idx_global = q_tile_start + ty; // The specific token index

  // Check bounds (if seq_len not divisible by Bc)
  bool active_q = (q_idx_global < seq_len);

  // Global Offsets
  int batch_idx = batch_head_idx / num_heads;
  int head_idx = batch_head_idx % num_heads;
  int base_offset = (batch_idx * num_heads * seq_len * head_dim) +
                    (head_idx * seq_len * head_dim);

  // Load Q into Register (If active)
  float q_val = 0.0f;
  if (active_q) {
      q_val = static_cast<float>(Q[base_offset + (q_idx_global * head_dim) + tx]);
  }

  // Initialize accumulators
  float m_i = -INFINITY;
  float l_i = 0.0f;
  float o_accum = 0.0f;

  // --------------------------------------------------------
  // 3. Main Loop over K/V Tiles
  // --------------------------------------------------------
  for (int j_base = 0; j_base < seq_len; j_base += Br) {
      
      // --- A. Collaborative Load into Shared Memory ---
      // We have Bc*head_dim threads. We need to load Br*head_dim elements.
      // 1D Linear index of this thread in the block
      int tid = ty * blockDim.x + tx; 
      
      // Iterate to cover the shared memory size
      for (int i = tid; i < Br * head_dim; i += blockDim.x * blockDim.y) {
          int row = i / head_dim;
          int col = i % head_dim;
          int k_idx_global = j_base + row;
          
          if (k_idx_global < seq_len) {
             K_tile[row * head_dim + col] = static_cast<float>(K[base_offset + k_idx_global * head_dim + col]);
             V_tile[row * head_dim + col] = static_cast<float>(V[base_offset + k_idx_global * head_dim + col]);
          } else {
             K_tile[row * head_dim + col] = 0.0f;
             V_tile[row * head_dim + col] = 0.0f;
          }
      }
      __syncthreads(); // Wait for Load

      // --- B. Compute Attention for this Tile ---
      if (active_q) {
          int valid_k = min(Br, seq_len - j_base);
          
          for (int k = 0; k < valid_k; ++k) {
              // 1. Dot Product (q_val * K_tile[k][tx])
              float k_val_s = K_tile[k * head_dim + tx];
              float partial = q_val * k_val_s;

              // --- REDUCTION START ---
              // Step 1: Reduce within the warp (Threads 0-31 sum up, 32-63 sum up, etc.)
              float score = warp_reduce_sum(partial); 
              
              // Step 2: Inter-Warp Reduction (if vector > 32)
              // We use shared memory to let warps communicate.
              if (head_dim > 32) {
                  int lane = tx % 32;
                  int warp = tx / 32;
                  int num_warps = head_dim / 32; 
                  
                  // Calculate where in red_smem this row (ty) should write
                  // Layout: [Row0_Warp0, Row0_Warp1... | Row1_Warp0...]
                  int smem_idx = (ty * num_warps) + warp;
                  
                  // A. Leaders write their partial sums to shared memory
                  if (lane == 0) {
                      red_smem[smem_idx] = score;
                  }
                  __syncthreads(); // Wait for all warps to write
                  
                  // B. Thread 0 of the row sums up the warp results
                  if (tx == 0) {
                      float total_score = 0.0f;
                      for(int w = 0; w < num_warps; ++w) {
                          total_score += red_smem[(ty * num_warps) + w];
                      }
                      // Write the TOTAL back to shared memory (broadcast source)
                      red_smem[ty * num_warps] = total_score;
                  }
                  __syncthreads(); // Wait for Thread 0 to sum
                  
                  // C. Everyone reads the total
                  score = red_smem[ty * num_warps];
              } else {
                  // If head_dim <= 32, the result is already in Lane 0. 
                  // Just broadcast it to the whole warp.
                  score = __shfl_sync(0xffffffff, score, 0);
              }
              // --- REDUCTION END ---

              score *= scale;

              // 2. Softmax Update
              float m_prev = m_i;
              m_i = fmaxf(m_i, score);
              float alpha = expf(m_prev - m_i);
              float beta = expf(score - m_i);
              l_i = (l_i * alpha) + beta;

              // 3. Output Update
              float v_val_s = V_tile[k * head_dim + tx];
              o_accum = o_accum * alpha + v_val_s * beta;
          }
      }
      __syncthreads(); // Wait before overwriting Shared Mem in next loop
  }

  // 4. Final Write
  if (active_q) {
      O[base_offset + (q_idx_global * head_dim) + tx] = static_cast<scalar_t>(o_accum / l_i);
  }
}

torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V) {
  auto O = torch::empty_like(Q);

  const int batch_size = Q.size(0);
  const int num_heads = Q.size(1);
  const int seq_len = Q.size(2);
  const int head_dim = Q.size(3);
  const float scale = 1.0f / sqrtf(head_dim);

  // 1. Grid Calculation
  // We process 'Bc' queries per block.
  // Total Queries = B * H * N
  int total_queries = batch_size * num_heads * seq_len;
  // Ceiling division to ensure we cover all queries
  int num_blocks = (total_queries + Bc - 1) / Bc;

  dim3 grid(num_blocks);

  // 2. Block Calculation (2D)
  // X dimension: Covers the vector size (head_dim)
  // Y dimension: Covers the number of queries in the tile (Bc)
  dim3 block(head_dim, Bc);

  // 3. Shared Memory Calculation
  // We need K + V + Reduction Buffer
  // Reduction Buffer needs: Bc rows * (head_dim / 32) floats per row
  // For head_dim=128, that's 4 floats per row.
  int num_warps_per_head = head_dim / 32; 
  size_t smem_size = (Br * head_dim * sizeof(float)) + 
                     (Br * head_dim * sizeof(float)) + 
                     (Bc * num_warps_per_head * sizeof(float));

  // 4. Launch
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      Q.scalar_type(), "flash_attention_kernel", ([&] {
        flash_attention_kernel_tiled<scalar_t><<<grid, block, smem_size>>>(
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