
### MP
intra-layer model parallel

#### PP
- layer-wise model parallelism
- https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/collective/collective_mp/pipeline.html

流水线并行Pipeline Parallel将模型按层分割到不同设备，形成处理流水线。传统PP为单mini-batch时序运行，同一时刻只有一个stage工作，效率低下

- PipeDream
- GPipe

#### TP
- Tensor Model Parallelism
- more general distributed tensor computation

张量并行Tensor Parallel将大型张量操作（如矩阵乘法）拆分到多个计算设备上执行，使得单个设备只需处理张量的一个子集，从而解决大模型训练中的内存和计算瓶颈问题。

1. Row-wise Parallelism
2. Column-wise Parallelism



#### CP
context parallel
- [context parallelism](context_parallelism.md)


