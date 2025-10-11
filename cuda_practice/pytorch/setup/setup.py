from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

#用一个pybind11 的 m 对象绑定所有cuda函数
setup(
    name='elementwise_ops_cuda',  # 这是你最终要 import 的模块名
    ext_modules=[
        CUDAExtension(
            name='elementwise_ops_cuda', # 模块名，与上面的 name 保持一致
            sources=[
                'main.cu',                      # 主绑定文件
                'element_wise/vector_add.cu',     # 加法实现
                'element_wise/vector_sub.cu',     # 减法实现
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })