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
  if(i & 0x80000000) {  //正数把符号位取回来 i& 011111...111
    return *reinterpret_cast<float*>(&(i &= ~0x80000000));
  }
  //是负数则按位取反
  return *reinterpret_cast<float*>(&(i = ~i));
}


//找到第k大的元素
__global__ void find_kth_value(const float* logits, size_t size, int k, float* result) {
  int tid = threadIdx.x;
  extern __shared__ int smem[]; //smem记录桶 相当于一个直方图

  unsigned int desired = 0; //确定位的具体值
  unsigned int desiredMask = 0;   //已经确定了哪些位
  int ktoFind = k;

  //从最高位开始迭代计算
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
        //digit记录value对应的桶
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
          desired |= static_cast<unsigned int>(i << pos); 
          desiredMask |= static_cast<unsigned int>(RADIX_MASK << pos);
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


// ======== 工具函数 =========
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << cudaGetErrorString(err) << ")\n";
        exit(EXIT_FAILURE);
    }
}

// ======== 主测试程序 =========
int main() {
    const int N = 1024;   // vocab_size
    const int K = 16;

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
    checkCuda(cudaMemset(d_count_gt, 0, sizeof(int)), "memset count_gt");
    checkCuda(cudaMemset(d_count_eq, 0, sizeof(int)), "memset count_eq");

    // 4. 调用 find_kth_value 内核
    int threads = THREAD_PER_BLOCK;
    int smem_size = (1 << 4) * sizeof(int);  // RADIX_SIZE = 16
    find_kth_value<<<1, threads, smem_size>>>(d_logits, N, K, d_kval);
    checkCuda(cudaDeviceSynchronize(), "find_kth_value sync");

    // 打印找到的阈值
    float h_kval;
    checkCuda(cudaMemcpy(&h_kval, d_kval, sizeof(float), cudaMemcpyDeviceToHost),
              "memcpy kval");
    std::cout << "[Threshold kth value] = " << h_kval << std::endl;

    // 5. 调用 count_topk 和 scatter_topk
    int blocks = (N + threads - 1) / threads;
    count_topk<<<blocks, threads>>>(d_logits, N, K, d_kval, d_topk_idx, d_count_gt);
    scatter_topk<<<blocks, threads>>>(d_logits, N, K, d_kval, d_topk_idx, d_count_gt, d_count_eq);
    checkCuda(cudaDeviceSynchronize(), "count_topk+scatter_topk sync");

    // 6. 拷贝回结果
    int h_count_gt = 0, h_count_eq = 0;
    checkCuda(cudaMemcpy(&h_count_gt, d_count_gt, sizeof(int), cudaMemcpyDeviceToHost),
              "memcpy count_gt");
    checkCuda(cudaMemcpy(&h_count_eq, d_count_eq, sizeof(int), cudaMemcpyDeviceToHost),
              "memcpy count_eq");

    std::vector<size_t> h_topk_idx(K);
    checkCuda(cudaMemcpy(h_topk_idx.data(), d_topk_idx, K * sizeof(size_t), cudaMemcpyDeviceToHost),
              "memcpy topk_idx");

    std::cout << "[Top-K indices count_gt=" << h_count_gt << ", count_eq=" << h_count_eq << "]\n";
    for (int i = 0; i < K; ++i) {
        std::cout << "top" << i << " idx=" << h_topk_idx[i]
                  << " val=" << h_logits[h_topk_idx[i]] << "\n";
    }

    // 7. 校验（CPU 对照）
    std::vector<float> sorted = h_logits;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    std::cout << "\n[CPU Top-K reference]:\n";
    for (int i = 0; i < K; ++i) std::cout << sorted[i] << " ";
    std::cout << std::endl;

    // 8. 清理
    cudaFree(d_logits);
    cudaFree(d_kval);
    cudaFree(d_topk_idx);
    cudaFree(d_count_gt);
    cudaFree(d_count_eq);

    return 0;
}