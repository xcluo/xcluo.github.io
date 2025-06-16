## PipeDream
> 论文：PipeDream: Fast and Efficient Pipeline Parallel DNN Training  
> MSR & Carnegie Mellon University & Stanford University, 2018 Jun

### 主要内容
- https://zhuanlan.zhihu.com/p/715442799
- https://www.cnblogs.com/rossiXYZ/p/15212165.html#%E6%BA%90%E7%A0%81%E8%A7%A3%E6%9E%90-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%B9%B6%E8%A1%8C%E4%B9%8Bpipedream1----profile%E9%98%B6%E6%AE%B5
- PipeDream aggressively pipelines minibatch processing, with different workers processing different inputs at any instant of time. This is accomplished by injecting multiple inputs into the worker with the first DNN layer, thereby keeping the pipeline full and ensuring concurrent processing on all workers  
- It also uses data parallelism for selected subsets of layers to balance computation load among workers.
- pipeline + MP + DP → PP
- PipeDream 通过运行简短的分析自动决定如何划分pipeline, using an algorithm that balances computation load among the different stages while minimizing communication
- PipeDream can use data parallelism for some stages—multiple workers can be assigned to a given stage, processing different minibatches in parallel.
- PipeDream在运行时会交织运行前向forward和后向backward
- 保证流水线时刻运行，不出现流水线停滞现象 while preventing excessive inprogress minibatches and ensuring model convergence
- Bulk Synchronous Parallel, BSP
- Stale Synchronous Parallel, SSP
- asynchronous parallel or ASP, reduces GPU idle time
- pipeline parallel将模型划分为多个部分，每部分叫做stage，每个stage对应一个gpu，其中输入部分是input stage，输出部分是output stage
- 传统的model-parallel DNN training results in severe under-utilization of GPU resources
- ![alt text](image.png)
- 为使每一时刻没有gpu闲置，通过inject multiple minibatches into the pipeline one after the other来避免该问题
- ![alt text](image-1.png)
- 由于通信时间为forward或backward的一小部分，又因为在pipeline中连续注入了多个minibatch，因此可以完美的避免通信等待
- ![alt text](image-2.png)
- PipeDream对DNN模型pipeline动态划分，期望：1）每个stage的计算量尽可能相等；2）使各stage间数据通信、传输的量越少越好。负载不均衡或者机器间过多的通信会降低效率，影响吞吐率

- PipeDream过程，给定$N$层，$M$个设备，首先在单个机器上运行一遍profiler，随后划分模型（同时确定replication factor以尽可能地较少训练时长）
    1. **profiling the DNN model**：DNN training shows little variance in the computation and communication time across minibatches (paper中使用了1000)，因此对每层记录3个数值{$T_l$: l的forward + backward计算时间综合；$a_l$：layer l输出激活值的size；$w_l$：layer l的参数的size；}
- 通信过程：1）发送方GPU→CPU；2）通过网络传输到接收方；3）接收方CPU→GPU。从layer l至layer l+1 激活值传输通信耗时为$C_l$，计算方式和为激活值数据量/带宽，可通过$a_l$估算。$W_l^m$ 表示layer l 在m个设备中DP同步参数权重的通信时长
- **PipeDream's Partitioning Algorithm**：Our partitioning algorithm takes the output of the profiling step, and computes: 1) a partitioning of layers into stages, 2) the ^^replication factor^^ for each stage, and 3) ^^optimal number of minibatches^^ to keep the training pipeline busy.
> 总共m个设备，可用于PP，也可用于DP，即 #DP + #PP = m
- 动态规划用于stage划分，$A(j, m)$ 表示前j层在m个设备上最优划分中的时长最大stage的耗时，$T(i\rightarrow j, m) = \frac{1}{m} \max \left( \sum_{l=i}^j T_l, \sum_{l=i}^j W_l^m \right)$ 表示层layer i 至 layer j在m个设备上DP花费的平均时长，因此会存在以下几种情况
    1. 只包含一个stage，DP m次

        $$
        A(j, m) = T(1\rightarrow j, m)
        $$

    2. layer 1至layer i进行$m-m^{'}$ PP，layer i+1 至 layer j 进行 $m^{'}$ DP

        $$
        A(j, m) = \min_{1 \le i \lt j} \min_{1 \le m^{'} \lt m} \max \begin{cases}
            A(i, m-m^{'}) \\
            2\cdot C_i \\
            T(i+1 \rightarrow j, m^{'})
        \end{cases}
        $$

        > $2\cdot C_i$ 中的2表示 forward activations + backward gradients

- **Initialization**：`A(1, m) = T(1 → 1, m) for m in range(1, M+1)`, `A(i, 1) = T(1 → i, 1) for i in range(1, N+1)`
- **Runtime Analysis**：A有N*M个子空间，因此为$O(NM)$，每个子空间需要分别对层i和数据并行数m进行分割遍历，为$O(NM)$，总复杂度为$O(N^2M^2)$

- 传统流水线并行采用全转发后全反向的调度，导致大量计算资源闲置，1F1B策略交错进行前向传播和后向传播，显著减少流水线气泡，每个工作器(worker)交替执行一个mini-batch的前向和一个mini-batch的后向

## PipeDream-2BW
> 论文：Memory-Efficient Pipeline-Parallel DNN Training  
> MSR & Stanford University, 2020 Jun, ICML 2021

### 主要内容
- PipeDream-2BW
- PipeDream-Flush