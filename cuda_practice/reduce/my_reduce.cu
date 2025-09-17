#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

const int BLOCK_SIZE = 1024;
#define FULL_MASK 0xffffffff

/* 你的 reduce0 kernel 保持不变 */
__global__ void reduce0(float* __restrict__ d_in,
                        float* __restrict__ d_out,
                        unsigned int N) {
    __shared__ float smem[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i  = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (i < N) ? d_in[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) d_out[blockIdx.x] = smem[0];
}

//解决空闲线程问题 让一半线程加载数据时进行一次加法
__global__ void reduce1(float* __restrict__ d_in,
                        float* __restrict__ d_out,
                        unsigned int N) {
    __shared__ float smem[BLOCK_SIZE];
    //相当于一个block处理原来两个block的数据
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int tid = threadIdx.x;
    float val = 0.0f;
    if (i < N) val += d_in[i];
    if (i + blockDim.x < N) val += d_in[i + blockDim.x];

    smem[tid] = val;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
        if (tid % (2 * s) == 0) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) d_out[blockIdx.x] = smem[0];
    
}

//解决warp发散问题 让同一个warp的thread执行相同的指令
__global__ void reduce2(float* __restrict__ d_in,
                        float* __restrict__ d_out,
                        unsigned int N) {
    __shared__ float smem[BLOCK_SIZE];
    //相当于一个block处理原来两个block的数据
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int tid = threadIdx.x;
    float val = 0.0f;
    if (i < N) val += d_in[i];
    if (i + blockDim.x < N) val += d_in[i + blockDim.x];

    smem[tid] = val;
    __syncthreads();

    for(unsigned int s = 1; s < blockDim.x; s <<= 1){
        int index = 2 * s *tid;
        //这个if循环会导致si'suo
        if(index < blockDim.x){ 
            smem[index] += smem[index+s];
        }
        __syncthreads();
    }

    if (tid == 0) d_out[blockIdx.x] = smem[0];
    
}

//解决warp发散问题同时解决 bank conflict
__global__ void reduce3(float* __restrict__ d_in,
                        float* __restrict__ d_out,
                        unsigned int N) {
    __shared__ float smem[BLOCK_SIZE];
    //相当于一个block处理原来两个block的数据
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int tid = threadIdx.x;
    float val = 0.0f;
    if (i < N) val += d_in[i];
    if (i + blockDim.x < N) val += d_in[i + blockDim.x];

    smem[tid] = val;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) d_out[blockIdx.x] = smem[0];
    
}

__device__ void warpReduce(float* cache, int tid) {
    int v = cache[tid] + cache[tid + 32];
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);
    cache[tid] = v;
}

//最后一个warp减少同步加循环展开
__global__ void reduce4(float* __restrict__ d_in,
                        float* __restrict__ d_out,
                        unsigned int N) {
    __shared__ float smem[BLOCK_SIZE];
    //相当于一个block处理原来两个block的数据
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int tid = threadIdx.x;
    float val = 0.0f;
    if (i < N) val += d_in[i];
    if (i + blockDim.x < N) val += d_in[i + blockDim.x];

    smem[tid] = val;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 32; s >>= 1) {
        if(tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    if(tid < 32) {
        warpReduce(smem, tid);
    }

    if (tid == 0) d_out[blockIdx.x] = smem[0];
    
}

__inline__ __device__ float blockReduce(float val) {
    const int tid = threadIdx.x;
    const int warpSize = 32;
    int lane = tid % warpSize;
    int warp = tid / warpSize;
    
    //先在每个warp内进行规约 一个block最多1024thread 32warp
#pragma unroll
    for(int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset, warpSize);
    }

    //一个warp规约一个block中的所有warp结果
    __shared__ float swarp[32];
    if(lane == 0) {
        swarp[warp] = val;
    }
    __syncthreads();

    if(warp == 0) {
        //不一定有1024个线程满32个warp
        val = (tid < blockDim.x / warpSize) ? swarp[tid] : 0.0f;
    }
#pragma unroll
    for(int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset, warpSize);
    }
    return val;
}

__global__ void reduce5(float* __restrict__ in,
                        float* __restrict__ out,
                        unsigned int N) {
    float sum = 0.0f;
    //idx是当前网格grid内的idx
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < N; i += gridDim.x * blockDim.x) {
        sum += in[i];
    }
    
    sum = blockReduce(sum);
    if(threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

/* ========== GPU 多级规约封装 ========== */
float reduce_gpu(const float* h_data, unsigned int N, int kernel_id)
{
    float* d_in  = nullptr;
    float* d_out = nullptr;

    // 分配一次最大所需显存：输入 + 中间结果
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, ((N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)) * sizeof(float));

    // 第一轮：把 host -> device
    cudaMemcpy(d_in, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // 多级规约
    float* src = d_in;
    unsigned int n = N;

    while (n > 1) {
        unsigned int blocks;

        if (kernel_id == 0) {
            // reduce0: 每个block处理 BLOCK_SIZE 个元素
            blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            reduce0<<<blocks, BLOCK_SIZE>>>(src, d_out, n);
        } else if (kernel_id == 1) {
            // reduce1: 每个block处理 BLOCK_SIZE * 2 个元素
            blocks = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
            reduce1<<<blocks, BLOCK_SIZE>>>(src, d_out, n);
        } else if(kernel_id == 2) {
            blocks = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
            reduce2<<<blocks, BLOCK_SIZE>>>(src, d_out, n);
        } else if(kernel_id == 3) {
            blocks = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
            reduce3<<<blocks, BLOCK_SIZE>>>(src, d_out, n);
        } else if(kernel_id == 4) {
            blocks = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
            reduce4<<<blocks, BLOCK_SIZE>>>(src, d_out, n);
        } else if(kernel_id == 5) {
            blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            reduce5<<<blocks, BLOCK_SIZE>>>(src, d_out, n);
        }

        // 下一轮输入
        src = d_out;
        n   = blocks;
    }

    // 把最终结果拷回 host
    float h_result;
    cudaMemcpy(&h_result, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    return h_result;
}


/* ========== CPU 验证 ========== */
float reduce_cpu(const std::vector<float>& v)
{
    //防止出现大数吃小数的问题
    return static_cast<float>(std::accumulate(v.begin(), v.end(), 0.0));
}

int main(int argc, char** argv)
{
    cudaDeviceReset(); // 重新启动 CUDA runtime
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " kernel_id (0 = kernel0, 1 = kernel1, 2 = kernel2 ....)\n";
        return 1;
    }
    int kernel_id = atoi(argv[1]);
    const unsigned int N = 32 * 1024 * 1024;
    std::vector<float> h_data(N, 1.0f);

    /* CPU 计时 */
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_result = reduce_cpu(h_data);
    auto cpu_end   = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = cpu_end - cpu_start;

    /* GPU 计时（重复 100 次取平均） */
    const int repeat = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* warm-up */
    reduce_gpu(h_data.data(), N, kernel_id);

    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) {
        volatile float dummy = reduce_gpu(h_data.data(), N, kernel_id);
        (void)dummy;              // 防止编译器把循环优化掉
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms_total = 0.0f;
    cudaEventElapsedTime(&gpu_ms_total, start, stop);
    float gpu_ms_avg = gpu_ms_total / repeat;

    /* 最终结果再跑一次，避免浮点累加误差 */
    float gpu_result = reduce_gpu(h_data.data(), N, kernel_id);

    /* 输出结果 */
    std::cout << "CPU result : " << cpu_result << '\n'
              << "CPU time   : " << cpu_ms.count() << " ms\n"
              << "GPU result : " << gpu_result << '\n'
              << "GPU avg    : " << gpu_ms_avg << " ms ("
              << repeat << " runs)\n"
              << "Speedup    : " << (cpu_ms.count() / gpu_ms_avg) << "x\n";

    if (std::abs(cpu_result - gpu_result) < 1e-5)
        std::cout << "Result verified successfully!\n";
    else
        std::cout << "Result verification failed!\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}