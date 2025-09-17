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

struct onlinePair {
    float maxval;
    float sumval;
};

// warp reduce max
// __device__ float warpReduceMax(float val) {
//     for(int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
//         val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
//     }
//     return val;
// }

// warp reduce sum
// __device__ float warpReduceSum(float val) {
//     for(int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
//         val += __shfl_down_sync(0xFFFFFFFF, val, offset);
//     }
//     return val;
// }

// __device__ onlinePair warpReduceOnline(onlinePair local)
// {
//     for(int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
//         float other_max = __shfl_down_sync(0xFFFFFFFF, local.maxval, offset);
//         float other_val = __shfl_down_sync(0xFFFFFFFF, local.sumval, offset);

//         float m = fmaxf(local.maxval, other_max);
//         float s = local.sumval * expf(local.maxval - m) + expf(other_val - m);

//         local.maxval = m;
//         local.sumval = s;
//     }

//     return local;
// }

__device__ onlinePair warpReduceOnline(onlinePair local)
{
    for(int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(0xFFFFFFFF, local.maxval, offset);
        float other_sum = __shfl_down_sync(0xFFFFFFFF, local.sumval, offset);

        float m = fmaxf(local.maxval, other_max);
        float s = local.sumval * expf(local.maxval - m) + other_sum * expf(other_max - m);

        local.maxval = m;
        local.sumval = s;
    }
    return local;
}


__global__ void online_softmax_kernel(float* out, const float* in, int N, int C) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_per_block = blockDim.x / WARP_SIZE;
    __shared__ float shared[WARP_SIZE*2]; // 足够存 maxvals 和 sumvals
    // extern __shared__ float input[];

    float* maxvals = shared;
    float* sumvals = &shared[warp_per_block];

    const float* input = in + bid * C;

    // for(int i = tid; i < C; i += blockDim.x) {
    //     input[i] = input[i];
    // }

    float maxval = -INFINITY;
    float sumval = 0.0f;

    for(int i = tid; i < C; i += blockDim.x) {
        float new_maxval = fmaxf(maxval, input[i]);
        sumval = sumval * expf(maxval - new_maxval) + expf(input[i] - new_maxval);
        maxval = new_maxval;
    }

    __syncthreads();

    //block内合并
    onlinePair p;
    p.maxval = maxval;
    p.sumval = sumval;

    // maxval = warpReduceMax(maxval);
    // sumval = warpReduceSum(sumval);
    //先在warp内合并
    onlinePair res = warpReduceOnline(p);


    if(lane_id == 0) {
        sumvals[warp_id] = res.sumval;
        maxvals[warp_id] = res.maxval;
    }
    __syncthreads();


    //thread0 合并所有warp
    if (tid == 0) {
        //warp之间的合并也必须严格按照online softmax合并
        float m = maxvals[0];
        float d = sumvals[0];

        for (int i = 1; i < warp_per_block; ++i) {
            float m_new = fmaxf(m, maxvals[i]);
            float d_new = d * expf(m - m_new) + sumvals[i] * expf(maxvals[i] - m_new);
            m = m_new;
            d = d_new;
        }

        maxvals[0] = m;
        sumvals[0] = d;
    }
    __syncthreads();    //需要同步 避免其他线程拿到还没修改的sum max

    float sum = sumvals[0];
    float max_val = maxvals[0];

    for(int i = tid; i < C; i += blockDim.x) {
        out[bid * C + i] = expf(input[i] - max_val) / sum;
    }
}

// ================= main =================
int main(int argc, char** argv) {
    cudaDeviceReset(); // 重新启动 CUDA runtime


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
    dim3 block(512);
    int repeat = 10;

    // int shmem_size = C * sizeof(float);

    // warmup
    for(int i = 0; i < repeat; ++i) {
        // online_softmax_kernel<<<grid, block, shmem_size>>>(d_out, d_in, N, C);
        online_softmax_kernel<<<grid, block>>>(d_out, d_in, N, C);
    }
    


    // timed loop
    cudaEventRecord(start);

    for(int i = 0; i < repeat; ++i) {
        // online_softmax_kernel<<<grid, block, shmem_size>>>(d_out, d_in, N, C);
        online_softmax_kernel<<<grid, block>>>(d_out, d_in, N, C);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    gpu_ms /= repeat;
    std::cout << "GPU Kernel  Avg Time (" << repeat << " runs): "
              << gpu_ms << " ms\n";

    // correctness
    float max_diff = 0.0f;
    float diff = 0.0f;
    int idx = 0;
    for (int i = 0; i < N*C; i++){
        max_diff = std::fmaxf(max_diff, fabs(h_out[i] - h_ref[i]));
    }

    for (int i = 0; i < N*C; i++){
        diff = fabs(h_out[i] - h_ref[i]);
        if(diff == max_diff) {
            idx = i;
            break;
        }
    }
    
        
    std::cout << "Max diff (GPU vs CPU) = " << max_diff << "\n";

    std::cout << "Max diff index : row " << idx / C  << " col " << idx % C << "\n";

    std::cout << "cpu : " << h_ref[idx] << " gpu " << h_out[idx] << "\n";

    std::cout << "Speedup: " << (cpu_ms.count() / gpu_ms) << "x\n";

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
    delete[] h_ref;

    return 0;
}