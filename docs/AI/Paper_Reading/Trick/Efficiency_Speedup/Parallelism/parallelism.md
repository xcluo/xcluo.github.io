分布式计算中的关键集体通信（Collective Communication）操作，用于在多个计算节点/设备间同步数据

- All-Reduce（全规约）：将多个设备上的数据通过某种操作聚合（如求和reduce_sum、求平均reduce_mean等），并将结果分发到所有设备
- All-Gather（全收集）：每个设备提供一部分数据，最终所有设备获得所有数据的完整拼接集合（如concat等）

分布式训练的通信后端  

- NCCL，电话网络
- Gloo，电子邮件
- MPI，及时通信软件

> 见论文Collective Communication，Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour


#### Hybrid Parallel
