## MoE
> 论文：Outrageously Large Neural Networks: the Sparsely-gated Mixture-of-Experts Layer  
> **MoE**：**M**ixture **o**f **E**xperts  
> Github: [mixture-of-experts](https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py#L17)  
> Google Brain & Jagiellonian University, ICLR 2017  

### 工作要点
### 主要内容
#### MoE工作原理
MoE通过门控网络加权$K_r$个专家网络的结果作为最终输出

1. 每次采样噪声以随机生成专家网络门限

    $$
    \begin{aligned}
        H(x)&=W_gx + noise*\text{softplus}(W_{noise}x) \\
        G(x)&=\text{softmax}\big(\text{KeepTopK}(H(x), K_r)\big) \\
    \end{aligned}
    $$

    > - $W_g \in \mathbb{R}^{N_r\times d}$ 与 $W_{noise} \in \mathbb{R}^{N_r\times d}$ 均使用全零初始化；随机值 $noise \in \mathbb{R}^{N_r}$ 
    > - $\text{KeepTopK}$ 函数对top-$K_r$ 的数值进行保留，其余值置为-$\infty$（经softmax后权重为0）
    > - $G(x)$ 具有稀疏性（$K_r \ll N_r$）与随机性（noise每次都随机采样）

2. 加权$K_r$个专家网络的结果作为最终输出

    $$
    y(x) = \sum_{i=1}^{N_r} G(x)_i*E_i(x) 
    $$

    > - 当$G(x)_i=0$时，可直接**不计算对应专家网络$E_i(x)$**以节省计算量
    
#### MoE Load Balance
1. $L_{importance}$ 通过**缩小专家网络权重值分数的离散程度**来均衡专家网络的使用率，该方法<span style="color: red">只均衡了专家网络总权重分数和</span>，而相同分数和可由少量高分数token或大量低分数token组合，因此<span style="color: red">可能会出现token number unblance现象</span>。


    $$
    \begin{aligned}
      Importance(X) \in \mathbb{R}^{N_r} =& \sum_{x\in X} G(x) \\
      CV =& \frac{\sigma}{\mu} \\
      L_{importance}(X) =& w_{importance}\cdot CV\big(Importance(X)\big)^2
    \end{aligned}
    $$

    > - 变异系数Coefficient of Variation，用于衡量数据集的相对离散程度。  

2. $L_{load}$ 通过**缩小token在各专家网络上分布的离散程度** 来均衡被专家网络被激活的概率

    $$
    \begin{aligned}
      kth\_excluding(v, k, i) =& \text{k-th highest component of v after excluding }v_i\\
      P(x, i) =& Pr\big(H(x)_i \gt kth\_excluding(H(x), K_r, i)\big) \\
      =& \Phi \bigg(\frac{(W_gx)_i - kth\_excluding(H(x), K_r, i)}{\text{softplus}((W_{noise}x)_i)}\bigg) \\
      Load(X)_i =& \sum_{x \in X} P(x, i) \\
      L_{load}(X) =& w_{load}\cdot CV\big(Load(X)\big)^2
    \end{aligned}
    $$

    > - $\Phi$ 为标准正态分布的累积分布函数CDF
    > - Eq 3. 通过移项和对称性$Pr(X \gt -x) = Pr(X \lt x)$求得，表示token在当前专家网络上的分布概率

#### Shrink Batch Problem
$\frac{K_rb}{N_r}\ll b$ inefficient as the $N_r$ increasing
- Mixing Data Parallelism and Model Parallelism
- Taking Advantage of Convolutionality
- Increasing Batch Size for a Recurrent MoE
- Network Bandwidth

#### Load Balance Loss
1. Expert-Level Balance Loss
2. Device-Level Balance Loss
3. Communication Balance Loss


- batchwise balance
  -  $M_{batch}(X, m)_{j, i}\begin{cases}
    1,\text{ if } X_{j, i} \text{ is the top } m \text{ values for expert } i \\
    0,\text{ otherwise }
  \end{cases}$
  - $L_{batchwise}$
- hierarchical MoEs in appendix B
#### 主要内容