#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#define WARP_SIZE 32

// CPU implementation
void softmax_forward_cpu(float *out, const float *in, int N, int C) {
  for (int i = 0; i < N; i++) {
    const float *in_row = in + i * C;
    float *out_row = out + i * C;

    float maxval = -INFINITY;
    for (int j = 0; j < C; j++) {
      if (in_row[j] > maxval) {
        maxval = in_row[j];
      }
    }
    float sum = 0.f;
    for (int j = 0; j < C; j++) {
      out_row[j] = expf(in_row[j] - maxval);
      sum += out_row[j];
    }
    float norm = 1.f / sum;
    for (int j = 0; j < C; j++) {
      out_row[j] *= norm;
    }
  }
}

// warp reduce max
__device__ float warpReduceMax(float val) {
    for(int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp reduce sum
__device__ float warpReduceSum(float val) {
    for(int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softmax_demo(float *out, const float *inp, int N,
                                        int C) {
  // out is (N, C) just like inp. Each row of inp will get softmaxed.
  // same as kernel3, but can handle any block size (multiple of 32)
  // each row of C elements is handled by block_size threads
  // furthermore, each block_size threads get executed in warps of 32 threads

  // special reduction operations warpReduceMax/warpReduceSum are used for
  // intra-warp reductions shared memory is used for inter-warp reduction
  extern __shared__ float shared[];
  int idx = blockIdx.x;
  int tid = threadIdx.x;
  int warpId = threadIdx.x / 32;  // warp index within a block
  int laneId = threadIdx.x % 32;  // thread index within a warp

  // the number of warps per block. recall that blockDim.x is block_size
  int warpsPerBlock = blockDim.x / 32;

  // shared[] must be allocated to have 2 * warpsPerBlock elements
  // first half for max values, the second half for sum values
  float *maxvals = shared;
  float *sumvals = &shared[warpsPerBlock];

  // one row of inp, i.e. inp[idx, :] of shape (C,)
  const float *x = inp + idx * C;

  // first, thread coarsening by directly accessing global memory in series
  float maxval = -INFINITY;
  for (int i = tid; i < C; i += blockDim.x) {
    maxval = fmaxf(maxval, x[i]);
  }
  // now within-warp reductions for maxval
  maxval = warpReduceMax(maxval);

  // the 0th thread of each warp writes the maxval of that warp to shared memory
  if (laneId == 0) maxvals[warpId] = maxval;
  __syncthreads();

  // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
  if (tid == 0) {
    float val = maxvals[tid];
    for (int i = 1; i < warpsPerBlock; i++) {
      val = fmaxf(val, maxvals[i]);
    }
    // store the final max in the first position
    maxvals[0] = val;
  }
  __syncthreads();
  // broadcast the max to all threads
  float offset = maxvals[0];

  // compute expf and write the result to global memory
  for (int i = tid; i < C; i += blockDim.x) {
    out[idx * C + i] = expf(x[i] - offset);
  }

  // okay now we calculated exp(x - max(x))
  // step 2: sum all the values and divide by the sum

  // thread coarsening for sum
  x = out + idx * C;
  float sumval = 0.0f;
  for (int i = tid; i < C; i += blockDim.x) {
    sumval += x[i];
  }
  // within-warp reduction for sumval
  sumval = warpReduceSum(sumval);

  // write sumval to shared memory
  if (laneId == 0) sumvals[warpId] = sumval;
  __syncthreads();

  // inter-thread reduction of sum
  if (tid == 0) {
    float val = sumvals[tid];
    for (int i = 1; i < warpsPerBlock; ++i) {
      val += sumvals[i];
    }
    sumvals[0] = val;
  }
  __syncthreads();
  // broadcast the sum to all threads
  float sum = sumvals[0];

  // divide the whole row by the sum
  for (int i = tid; i < C; i += blockDim.x) {
    out[idx * C + i] = x[i] / sum;
  }
}

// softmax kernel
__global__ void softmax_kernel1(float* out, const float* in, int N, int C) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int warp_per_block = blockDim.x / WARP_SIZE;
    __shared__ float shared[WARP_SIZE*2]; // 足够存 maxvals 和 sumvals
    float* maxvals = shared;
    float* sumvals = &shared[warp_per_block];

    const float* x = in + bid * C;

    // step1: reduce max
    float maxval = -INFINITY;
    for(int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    maxval = warpReduceMax(maxval);
    if(lane_id == 0) {
        maxvals[warp_id] = maxval;
    }
    __syncthreads();

    if(tid == 0) {
        float val = maxvals[0];
        for(int i = 1; i < warp_per_block; ++i) {
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();
    float offset = maxvals[0];

    // step2: exp(x - max)
    float* y = out + bid * C;
    for(int i = tid; i < C; i += blockDim.x) {
        y[i] = expf(x[i] - offset);
    }

    // step3: reduce sum
    float sumval = 0.0f;
    for(int i = tid; i < C; i += blockDim.x) {
        sumval += y[i];
    }
    sumval = warpReduceSum(sumval);
    if(lane_id == 0) {
        sumvals[warp_id] = sumval;
    }
    __syncthreads();

    if(tid == 0) {
        float val = sumvals[0];
        for(int i = 1; i < warp_per_block; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    float sum = sumvals[0];

    // step4: normalize
    for(int i = tid; i < C; i += blockDim.x) {
        y[i] /= sum;
    }
}

__global__ void softmax_kernel2(float* out, const float* in, int N, int C) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_per_block = blockDim.x / WARP_SIZE;
    __shared__ float shared[WARP_SIZE*2]; // 足够存 maxvals 和 sumvals
    extern __shared__ float sdata[];

    float* maxvals = shared;
    float* sumvals = &shared[warp_per_block];

    for(int i = tid; i < C; i += blockDim.x) {
        sdata[i] = in[i];
    }

    // step1: reduce max
    float maxval = -INFINITY;
    for(int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, sdata[i]);
    }
    maxval = warpReduceMax(maxval);
    if(lane_id == 0) {
        maxvals[warp_id] = maxval;
    }
    __syncthreads();

    if(tid == 0) {
        float val = maxvals[0];
        for(int i = 1; i < warp_per_block; ++i) {
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();
    float offset = maxvals[0];
    float sumval = 0.0f;
    // step2: exp(x - max) and recude sum
    for(int i = tid; i < C; i += blockDim.x) {
        sdata[i]= expf(sdata[i] - offset);
        sumval += sdata[i];
    }

    sumval = warpReduceSum(sumval);
    if(lane_id == 0) {
        sumvals[warp_id] = sumval;
    }
    __syncthreads();

    if(tid == 0) {
        float val = sumvals[0];
        for(int i = 1; i < warp_per_block; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }

    float sum = sumvals[0];

    // step4: normalize
    for(int i = tid; i < C; i += blockDim.x) {
        sdata[i] /= sum;
        out[bid * C + i] = sdata[i];
    }
}

// ================= main =================
int main(int argc, char** argv) {
    cudaDeviceReset(); // 重新启动 CUDA runtime
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " kernel_id (0 = demo, 1 = kernel1, 2 = kernel2)\n";
        return 1;
    }
    int kernel_id = atoi(argv[1]);

    int N = 32;
    int C = 4096;
    size_t size = N * C * sizeof(float);

    float* h_in  = new float[N*C];
    float* h_out = new float[N*C];
    float* h_ref = new float[N*C];

    for (int i = 0; i < N*C; ++i)
        h_in[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;

    // CPU baseline
    auto cpu_start = std::chrono::high_resolution_clock::now();
    softmax_forward_cpu(h_ref, h_in, N, C);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = cpu_end - cpu_start;
    std::cout << "CPU Time (1 run): " << cpu_ms.count() << " ms\n";

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 grid(N);
    dim3 block(128);
    int repeat = 1000;

    // warmup
    if (kernel_id == 1)
        softmax_kernel1<<<grid, block>>>(d_out, d_in, N, C);
    else if(kernel_id == 2) {
            int shmem_size = (2*(block.x/32) + C) * sizeof(float);
            softmax_kernel2<<<grid, block, shmem_size>>>(d_out, d_in, N, C);
        }
    else
        softmax_demo<<<grid, block, 2*(block.x/32)*sizeof(float)>>>(d_out, d_in, N, C);
    cudaDeviceSynchronize();

    // timed loop
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        if (kernel_id == 1)
            softmax_kernel1<<<grid, block>>>(d_out, d_in, N, C);
        else if(kernel_id == 2) {
            int shmem_size = (2*(block.x/32) + C) * sizeof(float);
            softmax_kernel2<<<grid, block, shmem_size>>>(d_out, d_in, N, C);
        }
        else
            softmax_demo<<<grid, block, 2*(block.x/32)*sizeof(float)>>>(d_out, d_in, N, C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    gpu_ms /= repeat;
    std::cout << "GPU Kernel" << kernel_id << " Avg Time (" << repeat << " runs): "
              << gpu_ms << " ms\n";

    // correctness
    float max_diff = 0.0f;
    for (int i = 0; i < N*C; i++)
        max_diff = std::max(max_diff, fabs(h_out[i] - h_ref[i]));
    std::cout << "Max diff (GPU vs CPU) = " << max_diff << "\n";

    std::cout << "Speedup: " << (cpu_ms.count() / gpu_ms) << "x\n";

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
    delete[] h_ref;

    return 0;
}