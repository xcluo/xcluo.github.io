## DeepCT
> 论文：Context-Aware Sentence/Passage Term Importance Estimation For First Stage Retrieval  
> DeepCT: **D**eep **C**ontextualized **T**erm weighting  
> Carnegie Mellon University 2019 Oct

### 主要内容
为改进传统静态词频统计方法（如TF-IDF、BM25），使用预训练语言模型基于语义动态**预测每个词**的重要性权重。

$$
w_i = \sigma (Wh_i + b)
$$

#### 数据处理

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^N(w_i - \hat{w}_i)^2
$$

> 监督训练需要词级权重标签（伪标签可能引入噪声）。

#### 动态加权改进
1. 基于加权词频的TF-IDF改进

    $$
    \begin{aligned}
        \text{wTF}(word, d) =& \sum w_{word}^{(d)} \\
        \text{wTF-IDF}(word, d) =& \text{wTF}(word, d)\cdot \text{IDF}(word)
    \end{aligned}
    $$

    > 计算 $\langle q, d\rangle$ 的加权TF-IDF向量相似度

2. 基于加权BM25的改进

    $$
    \begin{aligned}
        \text{wTF}_\text{BM25}(word, d) =& \sum \frac{w_{word}\cdot (k_1 + 1)}{w_{word} + k_1\cdot (1-b + b\frac{\vert d \vert}{\text{avg dl}})} \\
        \text{wBM25}(q, d) =& \sum_{word \in q} \text{IDF}(word)\cdot \text{wTF}_\text{BM25}(word, d)
    \end{aligned}
    $$

    > 保留BM25的非线性饱和特性，同时引入语义权重。

3. 基于稠密检索的扩展
    - 文档表示 $E_d = \sum_i w_i \cdot h_i$，问题查询表示 $E_q = \sum_i w_i \cdot h_i$  
    - 计算相似度，如余弦相似度，向量乘积等