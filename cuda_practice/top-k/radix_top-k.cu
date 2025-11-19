#include <float.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <limits>
#include <numeric>

#define THREAD_PER_BLOCK 512
#define WARP_SIZE 32
/* top-k算子实现思路
输入 logits[N]
         │
         ▼
[1] find_kth_value
   ├── 把 float 转成 radix-int
   ├── 从高位到低位循环 (每次 4 bits)
   │     ├── 清空桶 smem[16]
   │     ├── 统计当前候选的桶分布
   │     ├── 找出第 k 大所在桶
   │     └── 更新 desired / mask
   └── 输出 threshold 值 kval[0]
         │
         ▼
[2] count_topk
   ├── 遍历所有元素
   ├── if(val > threshold)
   │       └── atomicAdd(count_gt)
   └── 输出大于阈值的索引
         │
         ▼
[3] scatter_topk
   ├── 遍历所有元素
   ├── if(val == threshold)
   │       ├── base = count_gt[0]
   │       ├── pos = atomicAdd(count_eq)
   │       └── 写 topk_idx[base+pos]
   └── 直到写满 k 个
*/


///使用更高效更牛逼的基数排序思路 先找kth data 再找满足条件的
//使用桶排序，按位分桶找 kth data
//算法思想见文章https://zhuanlan.zhihu.com/p/1924510640476762596
const int RADIX_BITS = 4;
const int RADIX_SIZE = 1 << RADIX_BITS;
const int RADIX_MASK = RADIX_SIZE - 1;  //1111..111


// struct Pair {
//   size_t index;
//   float value;
// };


//float to int  正数符号位反转 负数全部按位取反
__device__ unsigned int float_to_radix_int(float val) {
  unsigned int i = *reinterpret_cast<unsigned int*>(&val);
  //负数按位取反
  if(i & 0x80000000) {
    return ~i;
  }
  
  return i | 0x80000000;  //正数反转符号位
}

__device__ float radix_int_to_float(unsigned int i) {
  //给的是i的地址
  if(i & 0x80000000) {  //正数把符号位取回来相当与 & 011111...111即~0x800000000
    return *reinterpret_cast<float*>(&(i &= ~0x80000000));
  }
  //是负数则按位取反
  return *reinterpret_cast<float*>(&(i = ~i));
}


//找到第k大的元素 有bug先不管
__global__ void find_kth_value(const float* logits, size_t size, int k, float* result) {
  int tid = threadIdx.x;
  extern __shared__ int smem[]; //smem记录桶 相当于一个直方图

  unsigned int desired = 0; //确定位的具体值
  unsigned int desiredMask = 0;   //已经确定了哪些位
  int ktoFind = k;

  //从最高位开始迭代计算 每次迭代 RADIX_BITS位 一共有1 << RADIX_BITS个桶
  for(int pos = sizeof(float) * 8 - RADIX_BITS; pos >= 0; pos -= RADIX_BITS) {
   
    //每一轮先清空直方图
    if(tid < RADIX_SIZE) {
      smem[tid] = 0;
    }

    //这里仍然需要同步，虽然只有0-15在清空，但是所有线程都在等着使用smem
    __syncthreads();

    for(int i = tid; i < size; i += blockDim.x) {
      unsigned int radix_val = float_to_radix_int(logits[i]);
      
      //上一轮也得满足条件 只统计最高位满足条件的
      //第一轮时 radix_val & 全0 = 全0
      if((radix_val & desiredMask) == desired) {
        //digit记录value对应的桶 val >> pos位
        int digit = (radix_val >> pos) & RADIX_MASK;
        atomicAdd(&smem[digit], 1);
      }
    }
    __syncthreads();

    if(tid == 0) {
      //从RADIX_SIZE-1开始即最高位开始
      for(int i = RADIX_SIZE - 1; i >= 0; --i) {
        if(smem[i] >= ktoFind) {
          //当前桶内的数量已经大于等于ktofind 则说明就在这个桶里面
          //此时说明第k大的元素就在这个桶里面 这个桶的二进制位就是满足条件的二进制位
          //前面右移pos位得到digit 所以这里要把桶的位左移pos 对desired进行修改
          //desired desiredMask 与操作上左移后的 i 和 RADIX_MASK
          desired |= static_cast<unsigned int>(i << pos); //i左移pos位和上一轮的desired |
          desiredMask |= static_cast<unsigned int>(RADIX_MASK << pos); //RADIX_MASK << pos 和上一轮的desiredMask | 
          break;
        }
        ktoFind -= smem[i];
      }
    }
    //这里同步是为了确保所有线程都看到更新后的desired 和 ktoFind
    __syncthreads();
  }
  

  //thread0 将结果写回
  if(tid == 0) {
    *result = radix_int_to_float(desired);
  }

}



__global__ void count_topk(const float* logits, size_t size, int k, float* kval, 
                              size_t* topk_idx, int* count_gt) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if(idx >= size)  return;

  float threshold = kval[0];

  float val = logits[idx];
  if(val > threshold) {
    //返回add之前的值
    int pos = atomicAdd(count_gt, 1);
    topk_idx[pos] = idx;
  }
}


__global__ void scatter_topk(const float* logits, size_t size, int k, float* kval, 
                              size_t* topk_idx, int* count_gt, int* count_eq) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if(idx >= size)  return;

  float threshold = kval[0];
  float val = logits[idx];
  if(val == threshold) {
    int base = count_gt[0];
    int pos = atomicAdd(count_eq, 1);
    if(base + pos < k) {
      topk_idx[base + pos] = idx;
    }
  }
}


__global__ void find_and_scatter_topk(const float* logits, size_t size, int k,
                                      float* kval, size_t* topk_idx,
                                      int* count_gt, int* count_eq) {
  int tid = threadIdx.x;
  
  // 定义所有需要的共享内存
  extern __shared__ int smem[];
  int* smem_hist = smem;
  float* s_kval = (float*)&smem_hist[RADIX_SIZE];
  int* s_count_gt = (int*)(s_kval + 1);
  int* s_count_eq = (int*)(s_count_gt + 1);
  
  // ===================== BUG 修复 =====================
  // 将状态变量移入共享内存
  unsigned int* s_desired = (unsigned int*)(s_count_eq + 1);
  unsigned int* s_desiredMask = s_desired + 1;
  int* s_ktoFind = (int*)(s_desiredMask + 1);
  // ====================================================

  // 阶段 1: find_kth_value (寻找阈值)
  if (tid == 0) {
    s_desired[0] = 0;
    s_desiredMask[0] = 0;
    s_ktoFind[0] = k;
  }
  __syncthreads(); // 确保所有线程看到初始化

  for(int pos = sizeof(float) * 8 - RADIX_BITS; pos >= 0; pos -= RADIX_BITS) {
    if(tid < RADIX_SIZE) {
      smem_hist[tid] = 0;
    }
    __syncthreads();

    // 所有线程都从共享内存读取正确的状态
    unsigned int current_desired = s_desired[0];
    unsigned int current_mask = s_desiredMask[0];

    for(int i = tid; i < size; i += blockDim.x) {
      unsigned int radix_val = float_to_radix_int(logits[i]);
      if((radix_val & current_mask) == current_desired) {
        int digit = (radix_val >> pos) & RADIX_MASK;
        atomicAdd(&smem_hist[digit], 1);
      }
    }
    __syncthreads();

    if(tid == 0) {
      for(int i = RADIX_SIZE - 1; i >= 0; --i) {
        if(smem_hist[i] >= s_ktoFind[0]) {
          s_desired[0] |= static_cast<unsigned int>(i << pos);
          s_desiredMask[0] |= static_cast<unsigned int>(RADIX_MASK << pos);
          break;
        }
        s_ktoFind[0] -= smem_hist[i];
      }
    }
    __syncthreads(); // 确保所有线程看到更新后的状态
  }
  
  if(tid == 0) {
    s_kval[0] = radix_int_to_float(s_desired[0]);
    s_count_gt[0] = 0;
    s_count_eq[0] = 0;
  }
  __syncthreads();

  // 阶段 2: count_topk (填充大于阈值的元素)
  float threshold = s_kval[0];
  for (unsigned int idx = tid; idx < size; idx += blockDim.x) {
    if (logits[idx] > threshold) {
      int pos = atomicAdd(s_count_gt, 1);
      if (pos < k) { // 添加边界检查，防止写入超过k个
          topk_idx[pos] = idx;
      }
    }
  }
  __syncthreads();

  // 阶段 3: scatter_topk (填充等于阈值的元素)
  int base = s_count_gt[0];
  for (unsigned int idx = tid; idx < size; idx += blockDim.x) {
    if (logits[idx] == threshold) {
      int pos = atomicAdd(s_count_eq, 1);
      if (base + pos < k) {
        topk_idx[base + pos] = idx;
      }
    }
  }
  __syncthreads();

  // 阶段 4: 写回最终结果
  if (tid == 0) {
    kval[0] = s_kval[0];
    count_gt[0] = s_count_gt[0];
    count_eq[0] = s_count_eq[0];
  }
}


// ======== 工具函数 =========
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << cudaGetErrorString(err) << ")\n";
        exit(EXIT_FAILURE);
    }
}

// ... (所有内核和 checkCuda 函数保持不变) ...

// ======== 主测试程序 =========
int main() {
    const int N = 1024 * 1024; // 使用更大的数据量以获得更明显的性能差异
    const int K = 128;
    const int BENCH_RUNS = 100; // 性能测试运行次数

    // 1. 生成随机数据
    std::vector<float> h_logits(N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-10.f, 10.f);
    for (auto &x : h_logits) x = dis(gen);

    // 2. 分配显存
    float *d_logits, *d_kval;
    size_t *d_topk_idx;
    int *d_count_gt, *d_count_eq;
    checkCuda(cudaMalloc(&d_logits, N * sizeof(float)), "malloc logits");
    checkCuda(cudaMalloc(&d_kval, sizeof(float)), "malloc kval");
    checkCuda(cudaMalloc(&d_topk_idx, N * sizeof(size_t)), "malloc topk_idx");
    checkCuda(cudaMalloc(&d_count_gt, sizeof(int)), "malloc count_gt");
    checkCuda(cudaMalloc(&d_count_eq, sizeof(int)), "malloc count_eq");

    // 3. 拷贝数据
    checkCuda(cudaMemcpy(d_logits, h_logits.data(), N * sizeof(float), cudaMemcpyHostToDevice),
              "memcpy logits");

    // 4. 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");
    float elapsed_unfused = 0.0f;
    float elapsed_fused = 0.0f;

    // ====================================================================
    // 5. 性能测试: 未合并的内核 (Unfused Kernels)
    // ====================================================================
    std::cout << "Benchmarking Unfused Kernels..." << std::endl;
    int threads = THREAD_PER_BLOCK;
    int blocks = (N + threads - 1) / threads;
    int smem_size_find = (1 << RADIX_BITS) * sizeof(int);

    // 预热
    find_kth_value<<<1, threads, smem_size_find>>>(d_logits, N, K, d_kval);
    count_topk<<<blocks, threads>>>(d_logits, N, K, d_kval, d_topk_idx, d_count_gt);
    scatter_topk<<<blocks, threads>>>(d_logits, N, K, d_kval, d_topk_idx, d_count_gt, d_count_eq);
    checkCuda(cudaDeviceSynchronize(), "Unfused warmup sync");

    checkCuda(cudaEventRecord(start), "record start unfused");
    for (int i = 0; i < BENCH_RUNS; ++i) {
        checkCuda(cudaMemset(d_count_gt, 0, sizeof(int)), "memset count_gt");
        checkCuda(cudaMemset(d_count_eq, 0, sizeof(int)), "memset count_eq");
        find_kth_value<<<1, threads, smem_size_find>>>(d_logits, N, K, d_kval);
        count_topk<<<blocks, threads>>>(d_logits, N, K, d_kval, d_topk_idx, d_count_gt);
        scatter_topk<<<blocks, threads>>>(d_logits, N, K, d_kval, d_topk_idx, d_count_gt, d_count_eq);
    }
    checkCuda(cudaEventRecord(stop), "record stop unfused");
    checkCuda(cudaEventSynchronize(stop), "sync stop unfused");
    checkCuda(cudaEventElapsedTime(&elapsed_unfused, start, stop), "elapsed time unfused");

    // ====================================================================
    // 6. 性能测试: 合并后的内核 (Fused Kernel)
    // ====================================================================
    std::cout << "Benchmarking Fused Kernel..." << std::endl;
    // int smem_size_fused = RADIX_SIZE * sizeof(int) + sizeof(float) + 2 * sizeof(int);
    int smem_size_fused = RADIX_SIZE * sizeof(int) + sizeof(float) + 2 * sizeof(int) + 2 * sizeof(unsigned int) + sizeof(int);

    // 预热
    find_and_scatter_topk<<<1, threads, smem_size_fused>>>(
        d_logits, N, K, d_kval, d_topk_idx, d_count_gt, d_count_eq);
    checkCuda(cudaDeviceSynchronize(), "Fused warmup sync");

    checkCuda(cudaEventRecord(start), "record start fused");
    for (int i = 0; i < BENCH_RUNS; ++i) {
        // 在融合内核内部会重置计数器，这里不需要 cudaMemset
        find_and_scatter_topk<<<1, threads, smem_size_fused>>>(
            d_logits, N, K, d_kval, d_topk_idx, d_count_gt, d_count_eq);
    }
    checkCuda(cudaEventRecord(stop), "record stop fused");
    checkCuda(cudaEventSynchronize(stop), "sync stop fused");
    checkCuda(cudaEventElapsedTime(&elapsed_fused, start, stop), "elapsed time fused");

    // ====================================================================
    // 7. 打印性能对比结果
    // ====================================================================
    std::cout << "\n=========== Performance Results ===========" << std::endl;
    std::cout << "Unfused (3 kernels) average time: " << elapsed_unfused / BENCH_RUNS << " ms" << std::endl;
    std::cout << "Fused (1 kernel) average time:    " << elapsed_fused / BENCH_RUNS << " ms" << std::endl;
    std::cout << "Speedup: " << elapsed_unfused / elapsed_fused << "x" << std::endl;
    std::cout << "=========================================\n" << std::endl;

    // ====================================================================
    // 8. 验证正确性 (使用最后一次融合内核的运行结果)
    // ====================================================================
    float h_kval;
    checkCuda(cudaMemcpy(&h_kval, d_kval, sizeof(float), cudaMemcpyDeviceToHost), "memcpy kval");
    std::cout << "[Threshold kth value] = " << h_kval << std::endl;

    int h_count_gt = 0, h_count_eq = 0;
    checkCuda(cudaMemcpy(&h_count_gt, d_count_gt, sizeof(int), cudaMemcpyDeviceToHost), "memcpy count_gt");
    checkCuda(cudaMemcpy(&h_count_eq, d_count_eq, sizeof(int), cudaMemcpyDeviceToHost), "memcpy count_eq");

    std::vector<size_t> h_topk_idx(K);
    checkCuda(cudaMemcpy(h_topk_idx.data(), d_topk_idx, K * sizeof(size_t), cudaMemcpyDeviceToHost), "memcpy topk_idx");

    std::cout << "[Top-K indices count_gt=" << h_count_gt << ", count_eq=" << h_count_eq << "]\n";
    for (int i = 0; i < K; ++i) {
        std::cout << "top" << i << " idx=" << h_topk_idx[i]
                  << " val=" << h_logits[h_topk_idx[i]] << "\n";
    }

    std::vector<float> sorted = h_logits;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    std::cout << "\n[CPU Top-K reference]:\n";
    for (int i = 0; i < K; ++i) std::cout << sorted[i] << " ";
    std::cout << std::endl;

    // 9. 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_logits);
    cudaFree(d_kval);
    cudaFree(d_topk_idx);
    cudaFree(d_count_gt);
    cudaFree(d_count_eq);

    return 0;
}
