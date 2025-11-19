/*
使用thread-tile访问优化 虽然每个线程负责不止一个计算，但是一个线程负责一个tile可以提高指令级并形程度
这样一个线程负责多个数据计算 MIO指令发射频率降低，提高指令级并形程度
在v2版本中，每个线程计算一个数据 计算访存比没有改变，一个线程处理多个tile,ILP(Instruction level paralle)增加
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>    // for fabsf
#include <fstream>  // for CSV output
#include <iostream>
#include <vector>

#define TOL 1e-5f

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

void checkCublasError(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}


//tile + shared 一个thead计算一个tile
//一个线程计算一个tile 提高指令并行度，减少对shared memeory的频繁读写
//主要变化在于一个线程负责一个数据计算变成一个线程负责一个tile(TM x TN)
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v3(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  
  int by = blockIdx.y;
  int bx = blockIdx.x;

  int thread_per_row = BN / TN;
  int thread_per_col = BM / TM;
  int thread_num = thread_per_col * thread_per_row;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A = &A[by * BM * K];
  B = &B[bx * BN];
  C = &C[by * BM * N + bx * BN];

  //计算shared memory中的二维索引
  int tx = (threadIdx.x % thread_per_row) * TN;
  int ty = (threadIdx.x / thread_per_row) * TM;

  int a_tile_row = threadIdx.x / BK;
  int a_tile_col = threadIdx.x % BK;
  int a_tile_stride = thread_num / BK;  //这是一个block一次能搬多少行

  int b_tile_row = threadIdx.x / BN;
  int b_tile_col = threadIdx.x % BN;
  int b_tile_stride = thread_num / BN;

  float tmp[TM][TN] = {0.f};
  for(int k = 0; k < K; k += BK) {
    float sum = 0.f;
    //分别加载A B矩阵数据到共享内存
    for(int i = 0; i < BM; i += a_tile_stride) {
      As[(i + a_tile_row) * BK + a_tile_col] = A[(i + a_tile_row) * K + a_tile_col];
    }
    for(int i = 0; i < BK; i += b_tile_stride) {
      Bs[(i + b_tile_row) * BN + b_tile_col] = B[(i + b_tile_row) * N + b_tile_col];
    }

    __syncthreads();

    A += BK;
    B += BK * N;
    for(int i = 0; i < BK; ++i) {
      for(int m = 0; m < TM; ++m) {
        for(int n = 0; n < TN; ++n) {
          tmp[m][n] += As[(ty + m) * BK + i] * Bs[i * BN + tx + n];
        }
      }
    }
    __syncthreads();
  }

  //每个线程把自己的结果存进去
  for(int m = 0; m < TM; ++m) {
    for(int n = 0; n < TN; ++n) {
      C[(ty + m) * N + tx + n] =
        alpha * tmp[m][n] + beta * C[(ty + m) * BN + tx + n];
    }
  }
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

std::vector<int> generateSizes() { return {128, 256, 512, 1024, 2048, 4096, 8192}; }
int main() {
  int device_id = 1;
  checkCudaError(cudaSetDevice(device_id), "cudaSetDevice failed");
  std::vector<int> sizes = generateSizes();

  // 打开CSV文件
  std::ofstream csv_file("sgemm_benchmark_v3_10_20.csv");
  csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

  for (int N : sizes) {
    std::cout << "Testing size: " << N << std::endl;

    size_t size = N * N * sizeof(float);
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C_cublas = (float *)malloc(size);
    float *C_v1 = (float *)malloc(size);

    float *d_A, *d_B, *d_C_v1;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed");

    bool out_of_memory = false;

    try {
      // 初始化矩阵 A 和 B
      for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
      }

      // 拷贝到设备
      checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy A to device failed");
      checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy B to device failed");

      cublasHandle_t handle;
      checkCublasError(cublasCreate(&handle), "cublasCreate failed");

      float alpha = 1.0f;
      float beta = 0.0f;

      cudaEvent_t start, stop;
      checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
      checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

      // warmup
      int warpup_time = 10;  // 热身次数
      for (int i = 0; i < warpup_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }
      cudaDeviceSynchronize();

      // cuBLAS SGEMM
      int repeat_time = 5;
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start cublas) failed");
      for (int i = 0; i < repeat_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }

      checkCudaError(cudaEventRecord(stop),
                     "cudaEventRecord(stop cublas) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize cublas failed");

      float cublas_time = 0;
      checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                     "cudaEventElapsedTime cublas failed");

      // 拷贝 cuBLAS 结果
      checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_cublas failed");

      // mysgemm_v3
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v3 failed");

      //一个线程块负责BMxBN
      //thread_num = 256是算出来的 thread_num等于row_threads * col_threads = BM/TM * BN/TN
      dim3 blockDim(256);
      dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

      for (int i = 0; i < warpup_time; ++i) {
        mysgemm_v3<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }

      cudaDeviceSynchronize();
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");

      for (int i = 0; i < repeat_time; ++i) {
        mysgemm_v3<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize v1 failed");
      float v1_time = 0;
      checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                     "cudaEventElapsedTime v1 failed");

      // 拷贝手写 kernel 结果
      checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_v1 failed");
      // 结果比较
      int error_count = 0;
      for (int i = 0; i < N * N && error_count < 10; ++i) {
        if (fabsf(C_cublas[i] - C_v1[i]) > TOL) {
          error_count++;
          // std::cout << "i " << i << std::endl;
        }
      }

      float cublas_gflops =
          repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);  // GFlops
      float v1_gflops =
          repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);  // GFlops
      // 写入CSV
      csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
               << (error_count == 0 ? "1" : "0") << std::endl;
      // std::cout << "error_count :" << error_count << std::endl;

      // 释放资源
      cublasDestroy(handle);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C_v1);

      free(A);
      free(B);
      free(C_cublas);
      free(C_v1);

    } catch (...) {
      std::cerr << "Out of memory or error during testing size: " << N
                << std::endl;
      out_of_memory = true;
    }

    if (!out_of_memory) {
      std::cout << "Finished size: " << N << std::endl;
    } else {
      csv_file << N << ",OOM,OOM,0" << std::endl;
    }
  }

  csv_file.close();

  std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v3_10_20.csv'"
            << std::endl;
  return 0;
}
