import torch
import elementwise_ops_cuda as elementwise_ops # 导入统一的模块

# 创建数据
a = torch.tensor([1., 2., 3.], device='cuda')
b = torch.tensor([10., 20., 30.], device='cuda')

# 调用函数
c = elementwise_ops.add(a, b)
d = elementwise_ops.sub(b, a)

print("a + b =", c)
print("b - a =", d)