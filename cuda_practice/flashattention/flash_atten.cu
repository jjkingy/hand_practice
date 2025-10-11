#include <stdio.h>
// #include "helper.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cassert>
#include <cmath>


/*******************辅助函数****************/
bool all_close(float *A, float *B, int m, int n) {
  for (int i = 0; i < m * n; i++) {
    if (fabs(A[i] - B[i]) > 1e-3f) {
      printf("A[%d] = %f, B[%d] = %f\n", i, A[i], i, B[i]);
      return false;
    }
  }
  return true;
}

// print matrix
void print_host_matrix(float *matrix, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f, ", matrix[i * n + j]);
    }
    printf("\n");
  }
}

void print_device_matrix(float *dev_ptr, int m, int n) {
  float *host_ptr = new float[m * n];
  cudaMemcpy(host_ptr, dev_ptr, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f, ", host_ptr[i * n + j]);
    }
    printf("\n");
  }
  free(host_ptr);
}


#define CUDA_CHECK(condition)                                          \
  do {                                                                 \
    cudaError_t error = condition;                                     \
    if (error != cudaSuccess) {                                        \
      printf(                                                          \
          "CUDA_CHECK error in line %d of file %s \
              : %s \n",                                                \
          __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError())); \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

// #define DEBUG
#ifdef DEBUG
#define DEBUG_BLOCK(expr) \
  do {                    \
    expr                  \
  } while (0)
#else
#define DEBUG_BLOCK(...) \
  do {                   \
  } while (0)
#endif

using FP = float;

const int Br = 2;
const int Bc = 2;

const int seq_len = 4;
const int dim = 16;

__global__ void naive_nrow_gemm(float *A, float *B, float *C, float a, float b,
                                int M, int N, int K, int mBlock);
__global__ void row_softmax(float *input, float *output, int n);
__global__ void naive_pv(float *P, float *V, float *O, int M, int N,
                         int mBlock);

__global__ void flash_attention_v2_kernel(FP *Q, FP *K, FP *V, FP *O,
                                          int seqlen, FP smScale);

void flash_attention_v2(float* Q, float* K, float* V, float* O, int seq_len, int dim) {
  float scale = 1.f / sqrtf(static_cast<float>(dim));

  int bs = 1;
  int head = 1;
  
  int Gc = 1;
  int Gr = (seq_len + Br - 1) / Br;
  dim3 grid = dim3(Gc, Gr);
  dim3 block = dim3(Bc, Br);

  flash_attention_v2_kernel<<<grid, block>>>(Q, K, V, O, seq_len, scale);
  DEBUG_BLOCK(printf("== v2: O ==\n"); print_device_matrix(O, seq_len, dim););

}

__global__ void flash_attention_v2_kernel(FP *Q, FP *K, FP *V, FP *O,
                                          int seq_len, FP scale) {
  //x y维度上的索引 后续处理dim也按照x y维度处理
  int groupTx = (dim + Bc - 1) / Bc;
  int groupTy = (dim + Br - 1) / Br;

  //q k v from global memory
  __shared__ float sQ[Br][dim];
  __shared__ float sK[Bc][dim];
  __shared__ float sV[Bc][dim];

  __shared__ float sQK[Br][Bc]; //缓存qk
  __shared__ float sO[Br][dim]; //缓存每一行的结果O
  __shared__ float sSafeE[Br][Bc]; // e^{x - max}

  __shared__ float sMax[Br];
  __shared__ float sDenom[Br];

  //***********下面还有很多共享内存没写，用到再写*********** */
  //temp 0
  // __shared__ float sO[]
  
  int tx = threadIdx.x; //[0, Bc]
  int ty = threadIdx.y; //[0, Br]
  int row = blockDim.y * blockIdx.y + ty; //当前是第几行

  if(row >= seq_len)  return;



  //load Q, O to shared memory
  for(int i = 0; i < groupTx; ++i) {
    sQ[ty][i * Bc + tx] = Q[row * dim + i * Bc + tx];
    sO[ty][i * Bc + tx] = 0;
  }
  sMax[ty] = -INFINITY;
  sDenom[ty] = 0;

  //在K V维度上迭代的次数
  int groupSeq = (seq_len + Bc - 1) / Bc;

  // load K, V block
  // Q[Br][dim] @ K[0..seqlen.step(Bc), dim]
  // compute partial sum of O[ty][dim] each iteration
  for(int j = 0; j < groupSeq; ++j) {
    if((j * Bc + tx) < seq_len) {
      for(int i = 0; i < groupTy; ++i) {
        sK[tx][i * Br + ty] = K[j * Bc * dim + tx * dim + i * Br + ty];
        sV[tx][i * Br + ty] = V[j * Bc * dim + tx * dim + i * Br + ty];
      }
    }

    __syncthreads();
    
    //compute qk
    float sum = 0.f;
    for(int i = 0; i < dim; ++i) {
      sum += sQ[ty][i] * sK[tx][i];
    }
    sQK[ty][tx] = sum * scale;
    __syncthreads();

    //online softmax
    float localMax = -INFINITY;
    for(int i = 0; i < Bc; i++) {
      localMax = fmaxf(localMax, sQK[ty][i]);
    }
    __syncthreads();
    float newMax = fmaxf(sMax[ty], localMax);

    sSafeE[ty][tx] = expf(sQK[ty][tx] - newMax);
    __syncthreads();

    float localDenom = 0.f;
    for(int i = 0; i < Bc; ++i) {
      localDenom += sSafeE[ty][i];
    }
    __syncthreads();
    
    float rescaleE = expf(sMax[ty] - newMax);  //后面还要用所以用寄存器存下来
    float newDenom = rescaleE * sDenom[ty] + localDenom;
    sMax[ty] = newMax;
    sDenom[ty] = newDenom;

    // NOTE:看下面注释 这里只处理加载进来的这块 先不要管整体
    // QK[Br, Bc] @ V[Bc, d] = O[Br, d]
    // tx in [0, Bc], ty in [0, Br]
    // slice-Bc and each O[ty, group.x] as accumulator
    for(int i = 0; i < groupTx; ++i) {  //这里的rescaleE是给分子乘的
      sO[ty][i * Bc + tx] = sO[ty][i * Bc + tx] * rescaleE;
      for(int k = 0; k < Bc; ++k) {
        sO[ty][i * Bc + tx] += sSafeE[ty][k] * sV[k][i * Bc + tx];
      }
    }
    __syncthreads();
  }
  //rescale O in the end
  for(int i = 0; i < groupTx; ++i) {
    //compute Oi and write to global
    //只在最后做一次rescale 避免中间softmax重复做scale/rescale
    O[row * dim + i * Bc + tx] = sO[ty][i * Bc + tx] / sDenom[ty];
  }

  //init max demon
}

void self_attention_cuda(float *Q, float *K, float *V, float *O, int m, int n) {
  int mBlock = 2;
  assert(m % mBlock == 0 && "mBlock should align");

  float sm_scale = 1.f / sqrtf(static_cast<float>(n));
  float *sm_o;
  cudaMalloc((void **)&sm_o, sizeof(float) * m * m);

  dim3 qk_block(m / mBlock, 1, 1);
  naive_nrow_gemm<<<1, qk_block>>>(Q, K, sm_o, sm_scale, 0, m, m, n, mBlock);
  cudaDeviceSynchronize();
  DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError()); printf("== naive QK ==\n");
              print_device_matrix(sm_o, m, m););

  // QK[M, M]
  dim3 sm_block(m, 1, 1);
  row_softmax<<<1, sm_block>>>(sm_o, sm_o, m);
  cudaDeviceSynchronize();
  DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
              printf("== naive softmax(QK) ==\n");
              print_device_matrix(sm_o, m, m););

  // QK[M, M] @ V[M, N]
  dim3 qkv_block(m / mBlock, 1, 1);
  naive_pv<<<1, qkv_block>>>(sm_o, V, O, m, n, mBlock);
  cudaDeviceSynchronize();
  DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
              printf("== naive softmax(QK)V ==\n");
              print_device_matrix(O, m, n););

  cudaFree(sm_o);
}

// naive gemm implement with slice-k
// perform C = aA@B + bC
// A[M, K] x B[K, N] = C[M, N]
// each thread process mblock rows of A
__global__ void naive_nrow_gemm(float *A, float *B, float *C, float a, float b,
                                int M, int N, int K, int mBlock) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // each thread process a range of rows
  idx *= mBlock;

  // A[mBlock, K] x B[N, K].T = C[mBlock, N]
  for (int i = idx; i < idx + mBlock; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[j * K + k];
      }
      // C[M, N]
      // C = aA@B + bC
      C[i * N + j] = a * sum + b * C[i * N + j];
    }
  }
}

// perform QK[M, M] @ V[M, N]
__global__ void naive_pv(float *P, float *V, float *O, int M, int N,
                         int mBlock) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // each thread process a range of rows
  idx *= mBlock;

  int K = M;
  // P[mBlock, M] x V[M, N] = O[mBlock, N]
  for (int i = idx; i < idx + mBlock; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.f;
      for (int k = 0; k < K; k++) {
        sum += P[i * K + k] * V[k * N + j];
      }
      // C[M, N]
      O[i * N + j] = sum;
    }
  }
}

// each thread process one row of softmax
__global__ void row_softmax(float *input, float *output, int n) {
  // assume id will not exceed row number of input
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  float max = -INFINITY;
  float sum = 0.f;

  // Find max
  for (int i = 0; i < n; i++) {
    if (input[idx * n + i] > max) {
      max = input[idx * n + i];
    }
  }

  // Compute numerator and denominator
  for (int i = 0; i < n; i++) {
    output[idx * n + i] = exp(input[idx * n + i] - max);
    sum += output[idx * n + i];
  }

  // Compute softmax
  for (int i = 0; i < n; i++) {
    output[idx * n + i] /= sum;
  }
}



void test_attention() {
  // seqlen
  int m = seq_len;
  // dim
  int n = dim;

  // Host pointer
  float *h_K = new float[m * n];
  float *h_Q = new float[m * n];
  float *h_V = new float[m * n];
  float *h_O = new float[m * n];
  float *h_O2 = new float[m * n];

  // 初始化 K, Q, V
  for (int i = 0; i < m * n; ++i) {
    h_K[i] = static_cast<float>(rand()) / RAND_MAX;
    h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
    h_V[i] = static_cast<float>(rand()) / RAND_MAX;

    DEBUG_BLOCK(h_K[i] = static_cast<float>(i); h_Q[i] = static_cast<float>(i);
                h_V[i] = static_cast<float>(i););
  }

  DEBUG_BLOCK(printf("== K ==\n"); print_host_matrix(h_K, m, n););

  float *d_K, *d_Q, *d_V, *d_O, *d_O2;
  // Malloc device memory
  cudaMalloc((void **)&d_K, sizeof(float) * m * n);
  cudaMalloc((void **)&d_Q, sizeof(float) * m * n);
  cudaMalloc((void **)&d_V, sizeof(float) * m * n);
  cudaMalloc((void **)&d_O, sizeof(float) * m * n);
  cudaMalloc((void **)&d_O2, sizeof(float) * m * n);

  // Copy data from host to device
  cudaMemcpy(d_K, h_K, sizeof(float) * m * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Q, h_Q, sizeof(float) * m * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, h_V, sizeof(float) * m * n, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  int warmup = 5;
  int repeat = 1000;
  //warm up
  for (int i = 0; i < warmup; i++) {
    // Launch kernel
    self_attention_cuda(d_Q, d_K, d_V, d_O, m, n);

    CUDA_CHECK(cudaGetLastError());
  }

  cudaEventRecord(start, 0);
  // Run test
  for (int i = 0; i < repeat; i++) {
    // Launch kernel
    self_attention_cuda(d_Q, d_K, d_V, d_O, m, n);

    CUDA_CHECK(cudaGetLastError());
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float self_atten_time = 0.f;
  cudaEventElapsedTime(&self_atten_time, start, stop);
  printf("Time for self_atten kernel execution: %.3f ms \n", self_atten_time / 100);

  // test flash attention 2
  //warm up
  for (int i = 0; i < warmup; i++) {
    flash_attention_v2(d_Q, d_K, d_V, d_O2, m, n);
    CUDA_CHECK(cudaGetLastError());
  }

  cudaEventRecord(start, 0);
  for (int i = 0; i < repeat; i++) {
    flash_attention_v2(d_Q, d_K, d_V, d_O2, m, n);
    CUDA_CHECK(cudaGetLastError());
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float flash_atten_time = 0.f;
  cudaEventElapsedTime(&flash_atten_time, start, stop);
  printf("Time for flash_atten_time kernel execution: %.3f ms \n", flash_atten_time / 100);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("speed up %.3f x", self_atten_time / flash_atten_time);

  // Result back to host
  cudaMemcpy(h_O, d_O, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_O2, d_O2, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
  bool res = all_close(h_O, h_O2, m, n);
  if (res) {
    printf("is equal\n");
  } else {
    printf("is not equal\n");
  }
  cudaFree(d_K);
  cudaFree(d_Q);
  cudaFree(d_V);
  cudaFree(d_O);
  cudaFree(d_O2);
  free(h_Q);
  free(h_K);
  free(h_V);
  free(h_O);
  free(h_O2);
}

int main() {
  // int epoch = 5;
  // for (int i = 0; i < epoch; i++) {
  //   test_attention();
  // }

  test_attention();

  return 0;
}