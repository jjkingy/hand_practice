#include <torch/extension.h>

// =======================================================
// 1. 函数声明
//    声明在其他 .cu 文件中定义的 C++ 包装函数。
//    这样链接器就能找到它们的实现。
// =======================================================
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
torch::Tensor vector_sub(torch::Tensor a, torch::Tensor b);


// =======================================================
// 2. 统一的模块定义
//    这是整个库唯一的 PYBIND11_MODULE。
//    TORCH_EXTENSION_NAME 将被 setup.py 中的 name 替换。
// =======================================================
PYBIND11_MODULE(elementwise_ops_cuda, m) {
    m.def(
      "add",                          // 在 Python 中调用的函数名
      &vector_add,                    // 指向要调用的 C++ 函数
      "Vector Add (CUDA)"             // 函数的文档字符串
    );
    m.def(
      "sub",                          // 在 Python 中调用的函数名
      &vector_sub,                    // 指向要调用的 C++ 函数
      "Vector Subtract (CUDA)"        // 函数的文档字符串
    );
}