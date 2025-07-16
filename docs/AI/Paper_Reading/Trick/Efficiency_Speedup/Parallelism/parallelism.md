- https://zhuanlan.zhihu.com/p/368828844

### 分布式训练架构
分布式计算中的关键集体通信（Collective Communication）操作，用于在多个计算节点/设备间同步数据

- All-Reduce（全规约）：将多个设备上的数据通过某种操作聚合（如求和reduce_sum、求loss平均reduce_mean等），并将结果分发到所有设备
- All-Gather（全收集）：每个设备提供一部分数据，最终所有设备获得所有数据的完整拼接集合（如concat等）

分布式训练的通信后端  

- NCCL，电话网络
- Gloo，电子邮件
- MPI，及时通信软件

> 见论文Collective Communication，Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour

### 参数更新策略


#### BSP
Bulk同步并行Bulk Synchronous Parallel是一种经典的并行机制，改模型将计算划分为多个超步（Supersteps），每个超步包含三个阶段

1. 本地计算（Computation）：所有machine并行执行本地计算
2. 全局通信（Communication）：machine间交互数据（如梯度同步）
3. 同步屏障（Barrier Synchronization）：所有的machine等待通信完成，确保一致性

关键特点："计算-通信-同步" 的严格交替执行，确保所有machine步调一致。因此超步时间为
$$
T = T_\text{compute} + T_\text{communicatoin} + T_\text{barrier}
$$

> - $T_\text{compute}$ 最慢machine的计算时间  
> - $T_\text{communicatoin}$ 最慢的通信延迟  
> - $T_\text{barrier}$ 同步开销  

!!! error ""
    木桶效应，因此同步开销会变大

#### ASP
异步并行Asynchronous Parallel是一种去中心化、无阻塞的并行机制，ASP的核心思想是允许所有计算节点machine独立运行，无需等待全局同步（局部优先，全局滞后），从而最大化硬件利用率，尤其适合大规模分布式训练和异构计算环境。

以参数服务器Parameter Server（PS）架构为例，ASP的计算步骤如下：

1. 从PS异步Pull参数（可能由于延迟或计算速度因素，当前machine的参数是过时版本 $W_{t-s_i}$）
2. 基于参数$W_{t-s_i}$，machine i使用本地数据计算梯度 $\nabla W_{t-s_i}$
3. 将梯度异步Push梯度到PS，PS立即更新全局参数 $W_{t+1} = W_t - \eta\cdot\nabla W_{t-s_i}$


#### SSP
旧版本同步并行Stale Synchronous Parallel是一种介于BSP严格同步和ASP完全异步的分布式并行计算机制，核心思想是通过容忍有限的计算延迟（Staleness Bound），在保证训练稳定性的同时，显著提高分布式系统的资源利用率。

以参数服务器Parameter Server（PS）架构为例，SSP的计算步骤如下：

1. 每个machine从PS异步Pull参数，并记录各自拉取的全局版本号 $t_i$
2. 基于本地参数$W_{t_i}$，machine i使用本地数据计算梯度 $\nabla W_{i_i}$  
3. 将梯度异步Push到PS，
    - if $t_i - \min \{t_j\}_{j=1}^N \le S$，则允许更新
    - else machine i暂停计算并等待其它machine追赶

4. PS聚合有效梯度并更新全局参数 $W_{t+1} = W_t - \eta\cdot\nabla W_{t_i}$
