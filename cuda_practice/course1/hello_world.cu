#include <cuda_runtime.h>

#include <iostream>
__global__ void hello_world(void) {
  printf("block idx:%d thread idx: %d\n", blockIdx.x, threadIdx.x);
  if (threadIdx.x == 0) {
    printf("GPU: Hello world!\n");
  }
}

int main(int argc, char **argv) {
  printf("CPU: Hello world!\n");
  hello_world<<<
      2, 6>>>();  
  cudaDeviceSynchronize();
  if (cudaGetLastError() != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError())
              << std::endl;
    return 1;
  } else {
    std::cout << "GPU: Hello world finished!" << std::endl;
  }
  std::cout << "CPU: Hello world finished!" << std::endl;
  return 0;
}