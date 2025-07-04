
### MP
intra-layer model parallel


#### TP
- Tensor Model Parallelism
- more general distributed tensor computation

张量并行Tensor Parallel将大型张量操作（如矩阵乘法）拆分到多个计算设备上执行，使得单个设备只需处理张量的一个子集，从而解决大模型训练中的内存和计算瓶颈问题。

1. Row-wise Parallelism
2. Column-wise Parallelism



#### CP
context parallel
- [context parallelism](context_parallelism.md)


