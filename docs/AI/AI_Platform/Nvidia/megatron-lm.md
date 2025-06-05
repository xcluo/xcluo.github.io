## Megatron-LM
> 论文：Megatron-LM: Training Multi-Billion Parameter Language Models Using
Model Parallelism  
> Github：[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)  
> Nvidia, 2019 Sep

### 主要内容
Megatron-LM是由NVIDIA开发的一个基于PyTorch的分布式训练框架，主要用于训练基于Transformer的大型语言模型。

- MLP first GEMM parallelism
- MLP second GEMM parallelism
- self-attention parallelism
- multihead self-attention parallelism with Q, K, V column parallel fashion, such that dealing with each attention head locally
- 将`W_Q, W_K, W_V` 按hidden_dim维度拆分至N个（N为#head）设备，即`W_Q = [W_Q_1, W_Q_2, ..., W_Q_N], W_K = [W_K_1, W_K_2, ..., W_K_N], W_V = [W_V_1, W_V_2, ..., W_V_N]`，$W\_{Q/K/V}\_i \in \mathbb{R}^{model\_dim \times head\_dim}$
- $W_O \in \mathbb{R}^{N*head\_dim \times model\_dim}$ 按行拆分`W_O = [W_O_1; W_O_2; ...; W_O_N]`，对各head的self-attention运算结果投影后进行all-reduce通信合并所有的结果
- 单层forward时2次all-reduce操作，backward时2次all-reduce操作
- 预测token阶段并行，本质上为GEMM parallelism，即hidden_state映射为token（$HE^T$，后者为embedding table）
- 对input embedding table的转置沿着column并行，实际上是沿着|V|方向拆分存储
- logits $HE^T = \text{GEMM}(Y_1, Y_2) = [XE_1^T, XE_2^T]$，使用all-reduce（sum）获得分母，进一步得到softmax_result.shape = (bs, seq_len, |V|)
- 避免all-gather操作的通信开销，仅通过all-reduce进行分母传递，从bs\*seq_len\*|V|的通信量减少为bs\*seq_len
- seq_length context parallelism
- BERT-style 的larger model训练使用pre-Norm较post-Norm更加有效
- To synchronize residual connection dropout across model parallel workers we seed
the random number generators at the beginning of training with the same seed.
- To achieve each worker to achieve randomness across the entire operation, we maintain a separate random number generator for dropout within model parallel regions. This random number generator is uniquely seeded for each model parallel worker.


## LLM using Megatron-LM
> 论文：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM  
> Nvidia & Stanford University & MSR, 2021 Apr, SC 2021