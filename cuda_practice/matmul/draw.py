import numpy as np
import matplotlib.pyplot as plt

# 读取 CSV 文件（跳过表头）
data = np.genfromtxt('/home/yy/Code/cuda_course/practice/matmul/sgemm_benchmark_v5_k8.csv', delimiter=',', skip_header=1)

# 提取三列数据
sizes = data[:, 0]          # Size
cublas_time = data[:, 1]    # CUBLAS_Time_ms
mygemm_time = data[:, 2]    # MySGEMM_v2_Time_ms

ratio = mygemm_time / cublas_time
print(cublas_time)
print(mygemm_time)
print(ratio)

# 创建图表
plt.figure(figsize=(10, 6))

# 绘图
plt.plot(sizes, cublas_time, label="cuBlas GFLOPS", marker='o', color='blue')
plt.plot(sizes, mygemm_time, label="MySGEMM GFLOPS", marker='s', color='orange')

# 设置图表标题和坐标轴标签
plt.title("CUBLAS vs sgemm_benchmark_v5_k8 GFLOPS", fontsize=14)
plt.xlabel("Matrix Size (N x N)", fontsize=12)
plt.ylabel("GFLOPS", fontsize=12)

# 设置 y 轴为对数刻度
# plt.yscale('log')

# 图例
plt.legend()

# 网格
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 自动调整布局
plt.tight_layout()

# 保存图像到文件（可选）
plt.savefig("cublas_vsmygemm_v5_k8.png", dpi=300)

# 显示图表
plt.show()
