import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def train(rank, world_size):
    # 1. 初始化分布式环境
    setup(rank, world_size)

    # 2. 定义一个简单的模型
    model = nn.Linear(10, 5).to(rank)
    # 关键步骤：使用DDP包装模型
    ddp_model = DDP(model, device_ids=[rank])

    # 3. 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # 4. 模拟训练过程
    for _ in range(10):
        # 模拟输入数据，每个GPU上的数据不同
        inputs = torch.randn(20, 10).to(rank)
        labels = torch.randn(20, 5).to(rank)

        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward() # DDP会自动进行梯度聚合
        optimizer.step()

        if rank == 0:
            print(f"Rank {rank}, Loss: {loss.item()}")

    # 5. 清理
    cleanup()

if __name__ == "__main__":
    # 假设我们有2个GPU
    world_size = 2
    # 在实际应用中，会使用 torch.multiprocessing.spawn 来启动多进程
    # 这里为了简化，只展示核心逻辑
    train(rank=0, world_size=2)
    train(rank=1, world_size=2)
    # 实际启动命令通常是：torchrun --nproc_per_node=2 your_script.py
    # 以下为概念性演示
    print("这是一个DDP的简化示例，实际运行需要通过torchrun或类似工具启动。")