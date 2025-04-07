#### Add-One Smoothing/Laplace Smoothing
加一平滑（又称拉普拉斯平滑），用于解决语言模型中零概率问题（即未登录词或低频词的概率估计），其核心思想是：**对所有可能的词频加1，确保没有词的概率为零**。

$$
P_{\text{Add-1}}(w\vert d) = \frac{c(w, d) + 1}{\vert d \vert + \vert V \vert }
$$

> - $c(w,d)$ 表示词 $w$  在文档 $d$ 中的出现次数（词频）
> - $\vert d\vert$ 表示文档 $d$ 的长度（总词数）
> - $\vert V \vert$ 表示词汇表大小（总单词数）

1. Add-k Smoothing，相对于Add-1增大平滑度，$k \gt 1$

    $$
    P_{\text{Add-k}}(w\vert d) = \frac{c(w, d) + k}{\vert d \vert + k\vert V \vert }
    $$

2. Lidstone Smoothing，相对于Add-1减小平滑度，$0 \lt k \lt 1$，适用于小规模数据集



#### Jelinek-Mercer Smoothing
Jelinek-Mercer平滑是信息检索和自然语言处理中广泛使用的语言模型平滑技术，通过线性插值（Linear Interpolation）将文档语言模型与背景语料语言模型结合，解决数据稀疏问题。其核心思想是：**信任文档中的词频统计，但对低频或未出现词，回退到全局背景模型**。

$$
P_{\text{JM}}(w\vert d) = \lambda \frac{c(w, d)}{\vert d \vert} + (1-\lambda)P(w\vert D)
$$

> - $P(w\vert D)$ 表示词 $w$ 在所有文档集合 $D$ 中的背景概率（即词 $w$ 在所有文档中的平均当初单位内出现次数），通常计算为 $P(w\vert D) = \frac{\sum_{d \in D} c(w,d)}{\sum_{d\in D} \vert d \vert}$
> - $0 \le \lambda \le 1$ 为插值系数，控制文档模型和背景模型的权重，经验取值[0.1, 0.7]


#### Dirichlet Prior Smoothing
迪利克雷先验平滑主要用于解决数据稀疏问题（即某些词在文档中未出现导致概率为零的情况），核心思想是**引入整个文档集合背景语料的词分布作为先验知识**，对文档中的词概率进行平滑调整，平滑后值域范围(0, 1]。

$$
P_{\text{Dir}}(w\vert d) = \frac{c(w, d) + \mu P(w|D)}{\vert d \vert + \mu}
$$

> - $\mu$ 为先验平滑参数，控制背景模型的权重，通常取值[1000, 2000]

由计算公式分母部分可知 $\sum_{w \in V} P_{\text{Dir}}(w\vert d) =1$，该平滑方法具有自适应特性，即长文档依赖数据（$\vert d \vert \gg \mu$），短文档依赖先验（$\vert d \vert \ll \mu$）