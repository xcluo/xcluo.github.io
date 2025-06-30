`Flash Attention` 是由Stanford AI Lab发表并实现的一些基于GPU底层架构实现Attention运算优化方案 (非近似算法)，其利用更少的GPU显存得到更快的运算的速率被许多企业和研究室采用，广泛应用于大多数LLM库

-----
- [Flash Attention v1](media/pdf/FlashAttention_v1.pdf)
- [Flash Attention v2](media/pdf/FlashAttention_v2.pdf)
- [Flash Attention v3](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)

### Standard Attention
#### 算法实现
标准attention算法实现

1. 从HBM分块加载$Q$、$K$，计算$S = QK^T$，将 $S$ 写回HBM
2. 从HBM读取$S$，计算 $P=\text{softmax}(S)$，将 $P$ 写回HBM
3. 从HBM分块加载$P$、$V$，计算$O=PV$，将 $O$ 写回HBM

!!! info ""
    - 当前主流的Attention加速算法都是近似算法（如稀疏Attention），以减少复杂度为$O(N^2d)$ Attention计算开销，但当前GPU的浮点数计算效率比IO访存效率快得多，*过分的优化FLOPS效果不明显*
    - 标准HBM IO访存复杂度=$O(Nd + N^2)$，<span style="color: red">瓶颈在HBM IO访存</span>。

### FlashAttention-1
通过`Q`、`K`、`V`分块tiling和重计算（不保存Attention的部分中间结果）的方式节省了HBM的IO访存次数

- [x] Attention部分显存使用数量级从平方降为线性
- [x] 最终以较少的GPU资源和更快的速度实现了与标准Attention一样的效果
- [x] 相同的GPU资源下能够训练更大的模型，取得更佳的效果表现


#### 动机
<div class="one-image-container">
    <img src="image/memory_hierarchy.png" style="width: 80%;">
    <figcaption>Attention存储访问示意图。（HBM：high bandwith memory）</figcaption>
</div>


===> **通过修改PyTorch或Tensorflow更底层CUDA逻辑实现IO访存的优化**

#### 基本原理
1. 分块
    - 将 `Q`分块为$T_r$个块，每个块维度$\in R^{B_r \times d}$
      > $N=B_r*T_r$
    - 将 `K`, `V`分块为$T_c$个块，每个块维度$\in R^{B_c \times d}$
      > $N=B_c*T_c$
    - 中间值 $S_{i, j}=Q_iK_j^T \in \mathbb{R}^{B_r\times B_c}$
    - 新增额外存储空间$m_i \in \mathbb{R}^{B_r}$ 表示实时存储i-th S块整合后各行最大值，临时变量$m_{i, j} \in \mathbb{R}^{T_r}$ 表示当前{i, j} S块的各行最大值
    - 中间值$P_{i, j} = \exp(S_{i,j} - \bar{m}_{i,j})$的局部指数减去当前块最大值的`exp`结果
    - 输出 `O` 分块为$T_r$ 个块，每个块维度$\in R^{B_r \times d}$
    - 新增额外存储空间$\ell \in \mathbb{R}^{T_r}$ 表示实时存储i-th P块整合后的exp_sum（除以了当前整合状态下最大值而归一化，防止累乘爆炸），临时变量$\ell_{i, j} \in \mathbb{R}^{T_r}$ 表示当前{i, j} P块的exp_sum（除以了当前块状态下最大值而归一化，防止累乘爆炸）
    - $O_i$表示`Q`分割得到$i \text{-} th$个块（即`Q_i`）的Attention分数


2. 重计算（不保存Attention的部分中间结果）
    - 不保存$S$至HBM，减少IO
    - 不保存$P$至HBM，减少IO

外循环是K、V，内循环是Q，前者只需要读2+T_r次，后者需要读1+2*T_c次
### FlashAttention-2

#### 动机
#### 基本原理

### FlashAttention-3
### 方法介绍

#### 基本原理



#### 底层优化方案
- [Flash Attention](Flash_Attention/FlashAttention.md)
    - v1：通过分块的方式节省了HBM的访存次数；在`batch_size`维度上并行
    - v2：减少了非矩阵乘法运算；修改`batch_size`维度并行至`seq_len`维度上并行；将分块的K、V矩阵快放入同一个thread block的不同warp中，通过共享内存的特性减少通信开销
    - v3：在Key和Value的序列长度`seq_len`上分块，进一步设置了多起点，从而新增了一个并行化操作。Q还是按照block顺序执行

<div class="one-image-container">
    <img src="image/flash_attention_v1,2_schematic_diagram.gif" style="width: 80%;">
    <figcaption>Flash Attention v1, v2运行原理图</figcaption>
</div>


<div class="one-image-container">
    <img src="image/flash_attention_v3_schematic_diagram.gif" style="width: 80%;">
    <figcaption>Flash Attention v3运行原理图</figcaption>
</div>

