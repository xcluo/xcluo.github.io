### MoE
> 论文：Outrageously Large Neural Networks: the Sparsely-gated Mixture-of-Experts Layer  
> **MoE**：**M**ixture **o**f **E**xperts  
> Github: [mixture-of-experts](https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py#L17)  
> Google Brain & Jagiellonian University,, ICLR 2017  

#### 工作要点
- 通过门控网络加权多个专家网络的结果作为最终输出

    $$
    \begin{aligned}
        y(x) &= \sum_{i=1}^n G(x)_i*E_i(x) \\
        G(x)&=\text{softmax}\big(\text{KeepTopK}(H(x), k)\big) \\
        H(x)&=xW_g + noise*\text{softplus}(xW_{noise})
    \end{aligned}
    $$

    !!! info ""
        - $K$ 为专家网络数
        - $G(x)\in \mathbb{R}^n$ 为**稀疏(k个值不为0)**门控网络，当$G(x)_i=0$时，可直接**不计算对应专家网络$E_i(x)$**
        - $\text{KeepTopK}(v, k)_i$ 表示$v_i$数值为$v$中的top-k就保留，否则置-$\infty$以使softmax权重为0
        - $noise\in \mathbb{R}^n$表示随机值，每次前向过程中都取一次随机值，尽可能地训练各个专家网络
        - $\text{softplus}(x)=\log(1+e^x)$

- 专家网络负载均衡，共同训练
    - L_importance
    - L_load in appendix A
    - init W_g and W_noise with all zeros

- batchwise balance
  -  $M_{batch}(X, m)_{j, i}\begin{cases}
    1,\text{ if } X_{j, i} \text{ is the top } m \text{ values for expert } i \\
    0,\text{ otherwise }
  \end{cases}$
  - L_batchwise
- hierarchical MoEs in appendix B
#### 主要内容