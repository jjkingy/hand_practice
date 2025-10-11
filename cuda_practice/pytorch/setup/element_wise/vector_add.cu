#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA 内核实现 (保持不变)
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    // ... existing code ...
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// C++ 包装函数 (保持不变)
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    // ... existing code ...
    TORCH_CHECK(a.is_cuda(), "Input tensor 'a' must be on a CUDA device");
    TORCH_CHECK(b.is_cuda(), "Input tensor 'b' must be on a CUDA device");
    auto c = torch::empty_like(a);
    int n = a.numel();
    const int threads_per_block = 256;
    const int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    vector_add_kernel<<<blocks_per_grid, threads_per_block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n
    );
    return c;
}

// 移除下面的 PYBIND11_MODULE 代码块
/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", ...);
}
*/