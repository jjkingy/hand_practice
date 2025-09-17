#include <cuda_runtime.h>
#include <iostream>


constexpr int BDIMX = 32;
constexpr int BDIMY = 8;

__global__ void naiveGmem(float* out, float* in, int nx, int ny) {
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
  if(ix < nx && iy < ny) {
    out[ix * ny + iy] = in[iy * nx + ix];
  }
}


// 调用核函数的封装函数
void call_naiveGmem(float *d_out, float *d_in, int nx, int ny) {
  dim3 blockSize(32, 8); // 线程块大小
  dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                (ny + blockSize.y - 1) / blockSize.y);
  naiveGmem<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);
}


__global__ void shared_transpose(float* out, float* in, int nx, int ny) {
  __shared__ float tile[BDIMY][BDIMX];
  
  //original index
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

  //linear global index for original
  unsigned int ti = iy * nx + ix;

  //下面要计算共享内存中的有关位置
  //index in transposed block(在一个块内的index)
  unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;

  //共享内存中的行列(这个时候已经按照转置之后的算了，blockDim.y现在是一行有多少列)
  unsigned int irow = bidx / blockDim.y;
  unsigned int icol = bidx % blockDim.y;

  //index in transposed matrix
  ix = blockIdx.y * blockDim.y + icol;
  iy = blockIdx.x * blockDim.x + irow;

  //linear global index for transposed matrix
  unsigned int to = iy * ny + ix;
  
  //这时候global index 和 shared index都计算好了
  if(ix < nx && iy < ny) {
    tile[threadIdx.y][threadIdx.x] = in[ti];
    __syncthreads();
    out[to] = tile[icol][irow];
  }

}

void call_shared_transpose(float* d_out, float* d_in, const int nx, const int ny) {
  dim3 block(BDIMX, BDIMY);
  dim3 grid((nx + BDIMX - 1) / BDIMX, (ny + BDIMY - 1) / BDIMY);

  shared_transpose<<<grid, block>>>(d_out, d_in, nx, ny);
}

__global__ void sharedpad_transpose(float* out, float* in, int nx, int ny) {
  const int pad = 1;

  //只是加一个pad 实际数据存储和访问没有变，加一个pad之后，随着icol变化，相邻每次间隔33个bank
  //这样一组连续的线程访问的bank路变为0, 1, 2, 3, ......
  __shared__ float tile[BDIMY][BDIMX + pad];
  
  //original index
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

  //linear global index for original
  unsigned int ti = iy * nx + ix;

  //下面要计算共享内存中的有关位置
  //index in transposed block(在一个块内的index)
  unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;

  //共享内存中的行列(这个时候已经按照转置之后的算了，blockDim.y现在是一行有多少列)
  unsigned int irow = bidx / blockDim.y;
  unsigned int icol = bidx % blockDim.y;

  //index in transposed matrix
  ix = blockIdx.y * blockDim.y + icol;
  iy = blockIdx.x * blockDim.x + irow;

  //linear global index for transposed matrix
  unsigned int to = iy * ny + ix;
  
  //这时候global index 和 shared index都计算好了
  if(ix < nx && iy < ny) {
    tile[threadIdx.y][threadIdx.x] = in[ti];
    __syncthreads();
    out[to] = tile[icol][irow];
  }

}

void call_sharedpad_transpose(float* d_out, float* d_in, const int nx, const int ny) {
  dim3 block(BDIMX, BDIMY);
  dim3 grid((nx + BDIMX - 1) / BDIMX, (ny + BDIMY - 1) / BDIMY);

  sharedpad_transpose<<<grid, block>>>(d_out, d_in, nx, ny);
}

int main() {
  int device_id = 0;
  cudaSetDevice(device_id);
  int nx = 4096;
  int ny = 4096;
  size_t size = nx * ny * sizeof(float);
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 主机内存分配
  float *h_in = (float *)malloc(size);
  float *h_out = (float *)malloc(size);

  // 初始化输入矩阵
  for (int i = 0; i < nx * ny; i++) {
    h_in[i] = float(int(i) % 11);
  }

  // 设备内存分配
  float *d_in, *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // 将数据从主机复制到设备
  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  //warm up
  int naive_repeat = 5;
  for(int i = 0; i < naive_repeat; i++) {
    call_naiveGmem(d_out, d_in, nx, ny);
  }

  naive_repeat = 10;
  // 调用核函数
  cudaEventRecord(start);
  for(int i = 0; i < naive_repeat; i++) {
    call_naiveGmem(d_out, d_in, nx, ny);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float naive_time = 0;
  cudaEventElapsedTime(&naive_time, start, stop);
  std::cout << "naive transpose总时间:" << naive_time / 10 << "ms" << std::endl;
  

  //warm up
  int shared_repeat = 5;
  for(int i = 0; i < shared_repeat; i++) {
    call_shared_transpose(d_out, d_in, nx, ny);
  }


  shared_repeat = 10;
  // 调用核函数
  cudaEventRecord(start);
  for(int i = 0; i < shared_repeat; i++) {
    call_shared_transpose(d_out, d_in, nx, ny);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float shared_time = 0;
  cudaEventElapsedTime(&shared_time, start, stop);
  
  std::cout << "shared unpad优化总时间:" << (shared_time / 10) << "ms" << std::endl;
  std::cout << "加速比: " << naive_time / shared_time << "x" << std::endl;

  // 将结果从设备复制回主机
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);


  //warm up
  int pad_repeat = 5;
  for(int i = 0; i < pad_repeat; i++) {
    call_naiveGmem(d_out, d_in, nx, ny);
  }

  pad_repeat = 10;
  // 调用核函数
  cudaEventRecord(start);
  for(int i = 0; i < pad_repeat; i++) {
    call_sharedpad_transpose(d_out, d_in, nx, ny);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float pad_time = 0;
  cudaEventElapsedTime(&pad_time, start, stop);
  std::cout << "shared pad总时间:" << pad_time / 10 << "ms" << std::endl;

  std::cout << "加速比: " << naive_time / pad_time << "x" << std::endl;

  // 将结果从设备复制回主机
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  // for (int j = 0; j < 10; ++j) {
  //   for (int i = 0; i < 10; ++i) {
  //     std::cout << h_in[j * nx + i] << " ";
  //   }
  //   std::cout << "\n";
  // }

  // printf("---------------\n");

  // for (int j = 0; j < 10; ++j) {
  //   for (int i = 0; i < 10; ++i) {
  //     std::cout << h_out[j * nx + i] << " ";
  //   }
  //   std::cout << "\n";
  // }

  // 释放内存
  free(h_in);
  free(h_out);
  cudaFree(d_in);
  cudaFree(d_out);

  std::cout << "Matrix transposition completed successfully." << std::endl;

  return 0;
}
