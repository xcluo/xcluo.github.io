### SubWord算法
SubWord算法，顾名思义就是把一个文本序列划分为更小的一个个**子词**（可能为单个字词的一部分，也可为常用的多个词）

#### SubWord分词器
1. [WordPiece](SubWord/subword_tokenize.md#wordpiece)
2. [UnigramUnigram Language Model (ULM)](SubWord/subword_tokenize.md#ulm)
3. [BPE (Byte Pair Encoding)](SubWord/subword_tokenize.md#bpe)

!!! info
    - pre_tokenizer用于split分割句子，bytelevel拆分单词至字节级，因此BPE字典中不会有中文（中文为多字节字符），**反而有很多无法理解的单字节串**
    - 为保证信息无损，BPE分词器会保留空格，因此需要预先将空格" "替换为一个不常用的字符，如`replace(" ", "Ġ")`，一般可以替换为一个词频少的char

#### 常用分词库
1. [SentencePiece](SubWord/SentencePiece.md)
2. [Tiktoken](https://github.com/openai/tiktoken)：使用Rust语言编写，高度优化，速度比基于Python的标准BPE快3-6倍，前者直接使用预定义的编码词表，后者可以自行构建词表

todo：Pinyin Tokenizer、拆字、繁简、字素(字组成结构)、OCR


[EM算法](https://cloud.tencent.com/developer/article/1608550)  

1. 通过实验结果和概率，找出最有可能导致这个结果的原因或者说参数，这个就叫做最大似然估计。  
2. Expectation Maximization algorithm  

    - 引入隐变量：即每个样本中都有一个无法得知隐藏信息的变量，需要（暴力）破解  
    - $\log P(X\vert \theta) = \log \sum_{Z} P(X, Z\vert \theta)$  
    - E step：计算各样本下各隐变量的期望概率，即填补缺失隐藏变量，$P(Z_A\vert X_i, \theta_A) = \frac{P(X_i\vert Z_A, \theta_A)P(Z_A)}{P(X_i\vert \theta_A)}$，满足$P(Z_A)$和$P(\theta_A)$互相独立  
    
        $$
        P(Z_A\vert X_i, \theta_A) = \frac{\theta_A^{H_i}(1-\theta_A)^{T_i}}{\sum_{z\in Z} \theta_z^{H_i}(1-\theta_z)^{T_i}}
        $$

        > $P(X\vert \theta)$ 表示所有$Z$的总概率  
        > $H_i$和$T_i$分别表示i-th样本中正面和反面次数

    - M step: 基于E步的结果重新估计$\theta$（极大似然估计），更新参数  

        $$
        \theta_A = \frac{\sum_{i=1}^n P(Z_A\vert X_i, \theta_A)H_i}{\sum_{i=1}^n P(Z_A\vert X_i, \theta_A)(H_i + T_i)}
        $$

    - 重复迭代，直至参数 $\theta$ 收敛
    - 基于结果反推隐变量概率，各投掷结果互相独立，概率计算公式为 $p^{H}(1-p)^{T}$
    - 该预测隐变量的方法是软分配soft assignment，Kmeas算法是硬分配，Kmeans是EM算法的一种简化版本