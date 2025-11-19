#include <cstdio>
#include <cuda_runtime.h>

__global__ void addKernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    int N = 1<<20;
    float *A, *B, *C;

    // 分配统一内存
    cudaMallocManaged(&A, N*sizeof(float));
    cudaMallocManaged(&B, N*sizeof(float));
    cudaMallocManaged(&C, N*sizeof(float));

    // 主机初始化
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // 启动内核
    addKernel<<<(N+255)/256, 256>>>(A, B, C, N);

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    // 验证结果
    printf("C[0] = %f\n", C[0]);

    // 释放内存
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}