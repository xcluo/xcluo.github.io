## SPLADE
> 论文：SPLADE: **SP**arse **L**exical **A**n**D** **E**xpansion Model for First Stage Ranking  
> 2021 Jul, SIGIR 2021  


### 主要内容
#### SparTerm
基于各token对词表中各词的语义相关性进行查询拓展

$$
\begin{aligned}
    w_{i, j} =& h_i^TE_j + b_j \\
    w_j =& \sum_{i=1}^{sel\_len} \log \big(1+ \text{ReLU}(w_{i, j})\big)
\end{aligned}
$$


#### Rank Loss
1. InfoNCE Loss负样本选择
    - in batch negatives
    - hard negative sample with high BM25

    $$
    \mathcal{L}_\text{rank-IBN} = - \log \frac{e^{s(q_i, d_i^+)}}{\sum_{j}e^{s(q_i, d_j^+)} + e^{s(q_i, d_i^-)}}
    $$

    > $s(q, d) = \langle w^{q}, w^{d} \rangle $


2. 防止词项权重分布出现长尾效应

    $$
    \begin{aligned}
        \mathcal{L}_\text{reg}^d =& \sum_{j=1}^{\vert V\vert} \bigg( \frac{1}{N}\sum_{i=1}^N w_j^{(d_i)} \bigg) ^ 2 \\
        \mathcal{L}_\text{reg}^q =& \sum_{j=1}^{\vert V\vert} \bigg( \frac{1}{N}\sum_{i=1}^N w_j^{(q_i)} \bigg) ^ 2 \\ 
    \end{aligned}
    $$

3. overall loss

    $$
    \mathcal{L} = \mathcal{L}_\text{rank-IBN} + \lambda_q\mathcal{L}_\text{reg}^q + \lambda_d\mathcal{L}_\text{reg}^d
    $$