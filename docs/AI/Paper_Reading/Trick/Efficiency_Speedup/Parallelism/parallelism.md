分布式计算中的关键集体通信（Collective Communication）操作，用于在多个计算节点/设备间同步数据

- All-Reduce（全规约）：将多个设备上的数据通过某种操作聚合（如求和reduce_sum、求平均reduce_mean等），并将结果分发到所有设备
- All-Gather（全收集）：每个设备提供一部分数据，最终所有设备获得所有数据的完整拼接集合（如concat等）

分布式训练的通信后端  

- NCCL，电话网络
- Gloo，电子邮件
- MPI，及时通信软件

> 见论文Collective Communication，Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour

### DP
https://www.cnblogs.com/gzyatcnblogs/articles/17946484#dataparallel

#### DP
- https://www.cnblogs.com/CircleWang/p/15620825.html
- https://www.cnblogs.com/gzyatcnblogs/articles/17946484#dataparallel

1. 将模型复制到各个GPU中，并将一个batch的数据划分为多个mini_batch并分发给多个GPU
2. 各个GPU独自完成mini_batch的forward，并把获得的output传递给GPU_0（主GPU）
3. GPU_0整合各个GPU传递过来的output，并计算loss（此时GPU_0可以对所有loss进行一些聚合操作）
4. GPU_0归并loss后，进行backward以及梯度下降后完成模型参数更新，随后将更新后模型传递给其它GPU

> 以上就是DP模式下多卡GPU进行训练的方式。其实可以看到GPU_0不仅承担了前向传播的任务，还承担了收集loss，并进行梯度下降。因此在使用DP模式进行单机多卡GPU训练的时候会有一张卡的显存利用会比其他卡更多，那就是你设置的GPU_0。  
> 每张GPU都保留了模型参数副本


```python
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,1' # GPU_0为卡2，GPU_1为卡3，GPU_3为卡1
device_ids = [0, 1, 2]

model = nn.DataParallel(model, device_ids=device_ids)
model = nn.DataParallel(model)               # 在执行该语句之前最好加上model.cuda(),保证模型存在GPU上
                                             # 内部batch的拆分被封装在了DataParallel模块中
```
> 由于我们的model被`nn.DataParallel()`包裹住了，所以如果想要储存模型的参数，需要使用model.module.state_dict()的方式才能取出，不能直接是model.state_dict()  

> 只有一个主进程，主进程下有多个线程，每个线程管理一个device的训练。因此DP内存中只存在一份数据，各个线程间共享该数据。  

> 仅限单机多卡
#### DDP
分布式数据并行Distributed Data Pparallelism，多个进程，每个进程会独立加载完整的数据并进行训练

- 读取不重叠的数据
- 支持跨机器


```python
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# 1. 基础模块 ### 
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        cnt = torch.tensor(0)
        self.register_buffer('cnt', cnt)

    def forward(self, x):
        self.cnt += 1
        # print("In forward: ", self.cnt, "Rank: ", self.fc.weight.device)
        return torch.sigmoid(self.fc(x))

class SimpleDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
# 2. 初始化我们的模型、数据、各种配置  ####
## DDP：从外部得到local_rank参数。从外面得到local_rank参数，在调用DDP的时候，其会自动给出这个参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

## DDP：DDP backend 通信后端初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

## 假设我们有一些数据
n_sample = 100
n_dim = 10
batch_size = 25
X = torch.randn(n_sample, n_dim)  # 100个样本，每个样本有10个特征
Y = torch.randint(0, 2, (n_sample, )).float()

dataset = SimpleDataset(X, Y)
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

## 构造模型
model = SimpleModel(n_dim).to(local_rank)
## DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))

## DDP: 构造DDP model —————— 必须在 init_process_group 之后才可以调用 DDP
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

## DDP: 要在构造DDP model之后，才能用model初始化optimizer。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_func = nn.BCELoss().to(local_rank)

# 3. 网络训练  ###
model.train()
num_epoch = 100
iterator = tqdm(range(100))
for epoch in iterator:
    # DDP：设置sampler的epoch，
    # DistributedSampler需要这个来指定shuffle方式，
    # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
    data_loader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    for data, label in data_loader:
        data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label.unsqueeze(1))
        loss.backward()
        iterator.desc = "loss = %0.3f" % loss
        optimizer.step()

    # DDP:
    # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
    #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
    # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
    if dist.get_rank() == 0 and epoch == num_epoch - 1:
        torch.save(model.module.state_dict(), "%d.ckpt" % epoch)
```


#### FSDP
Fully Sharded Data Parallel，全切片数据并行，结合了ZeRO思路进行了参数、梯度和优化期状态分片


### MP
intra-layer model parallel

#### PP
- Pipeline Model Parallelism
- https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/collective/collective_mp/pipeline.html

流水线并行Pipeline Parallel将模型按层分割到不同设备，形成处理流水线。传统PP为单mini-batch时序运行，同一时刻只有一个stage工作，效率低下

- PipeDream
- GPipe

#### TP
- Tensor Model Parallelism

张量并行Tensor Parallel将大型张量操作（如矩阵乘法）拆分到多个计算设备上执行，使得单个设备只需处理张量的一个子集，从而解决大模型训练中的内存和计算瓶颈问题。

1. Row-wise Parallelism
2. Column-wise Parallelism

- Megatron-LM
- Colossal-AI


#### CP
context parallel
- [context parallelism](context_parallelism.md)

#### Hybrid Parallel
