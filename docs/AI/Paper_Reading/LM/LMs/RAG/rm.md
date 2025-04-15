Relevance Model，两次检索：  

1. 基于query从知识库检索top-k文档；  
2. 基于top-k文档统计结果获取top-M拓展词，整合 $q_\text{expanded} = \text{concat}\big(q, RM(D_R, M)\big)$ 中进行二次检索

#### RM1
Relevance-Based Language Models

1. 基于[Dirichlet平滑](../../../../Metrics/smoothing_metrics.md#dirichlet-prior-smoothing) 或 [Jelinek-Mercer平滑](../../../../Metrics/smoothing_metrics.md#jelinek-mercer-smoothing) 计算词在文档中的概率获取伪相关文档的加权平均概率 $P(w\vert R) = \frac{1}{\vert D_R \vert}\sum_{d\in D_{R}} P(w\vert d)$

> - 如果初始top-k 文档不相关，拓展词可能引入噪声  
> - 未考虑词权重，所有文档平均加权，可能受高频词干扰  

#### RM2
A Generative Theory of Relevance，不再假设文档中的词互相独立，而是考虑词项之间的依赖关系（如二元模型），从而更准确地估计相关性模型

$$
\begin{aligned}
    P(w\vert R) =& \sum_{d \in D_R} P(w\vert d)\times P(q\vert d)\times P(d\vert R) \\
    =& \frac{1}{\vert D_R \vert} \sum_{d \in D_R} P(w\vert d)\times P(q\vert d) \\
    P(w\vert d) \approx& P(w\vert w_\text{prev}, d) \\
    P(q\vert d) =& \prod_{w \in q} P(w\vert d) \\
    P(d\vert R) =& \frac{1}{\vert D_R \vert}
\end{aligned}
$$

> - $P(w\vert d)$ 词 $w$ 在文档 $d$ 中的概率（考虑词依赖，如二元模型）
> - $P(q\vert d)$ 查询 $q$ 在文档 $d$ 中的生成概率
> - $p(d\vert R)$ 文档 $d$ 的相关性概率
> - 计算复杂度高：需维护词共现统计

#### RM3
UMass at TREC 2004: Novelty and HARD，通过控制原始查询和拓展词的权重减少噪声词的影响

$$
\begin{aligned}
P(w \vert q_\text{expanded}) =& \lambda P(w\vert q) + (1-\lambda) \sum_{d \in D_R}P(w\vert d)P(d\vert q)   \\
P(d\vert q) =&  \text{BM25}(q, d)
\end{aligned}
$$

> - $\lambda \in [0, 1]$，插值系数，经验取值 0.5~0.7  
> - 一元模型

#### RM4
Adaptive Relevance Feedback in Information Retrieval，进一步引入了负反馈样本

$$
\begin{aligned}
    P(w \vert q_\text{expanded}) =& \lambda P(w\vert q) + (1-\lambda) \bigg[\sum_{d \in D_R} P(w\vert d)P(d\vert q) -\alpha \sum_{d \in D_{NR}}P(w\vert d)P(d\vert q) \bigg]
\end{aligned}
$$

> - 使用了分数排名靠后的 $D_{NR}$ 低相关文档进行负反馈，经验取后100~200文档  
> - $\alpha$ 负反馈权重，经验取值 0.1~0.5  


#### Rocchio
用户反馈算法，Rocchio算法

$$
q_\text{new} = \alpha q + \beta \frac{1}{\vert D_R \vert} \sum_{d \in D_R} \overrightarrow{h(q, d)} - \gamma \frac{1}{\vert D_{NR} \vert} \sum_{d \in D_{NR}} \overrightarrow{h(q, d)}
$$

> - $\alpha, \beta, \gamma$ 相关性权重参数，控制各部分参数，通过人户反馈的人工标注数据进行调优
