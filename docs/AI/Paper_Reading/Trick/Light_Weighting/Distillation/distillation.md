### 蒸馏技巧

#### Soft Label & Hard Label
教师模型的分类结果一般为连续概率软标签，可只使用软/硬标签蒸馏或利用混合标签进行蒸馏  


1. **Soft Label**：$\mathcal{L}_\text{soft} = \sum_{i}\sum_{k} p_{i, k}^\text{teacher}\log q_{i, k}$
2. **Hard Label**：$\mathcal{L}_\text{hard} = \sum_{i} y_i^\text{teacher}\log q_{i}$
3. **Hybrid Label**: $\mathcal{L} = \alpha\mathcal{L}_\text{soft} + (1-\alpha) \mathcal{L}_\text{hard}$，$\alpha$ 为权重超参数

!!! info ""
    - 蒸馏时可使用预训练 word_embedding或model 进行快速迁移学习
    - 挑选蒸馏样本时，可选择教师模型命中的正样本$N_\text{pos}$，并采样收集负样本 $pp*N_\text{neg}$
    - 为使学生模型充分利用训练数据，可将教师模型的训练数据加入作为蒸馏数据集


#### Temperature
温度系数用于控制教师模型的软标签概率分布 $p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$，

- $T\rightarrow 0$，退化为硬标签蒸馏
- $T\lt 1$，锐化概率分布，使主要类别概率更显著，一般不适用而是直接应用硬标签
- $T=1$，退化为标准softmax
- $T\gt 1$，软化概率分布，让次要类别概率更显著，==使用较多==
- $T\rightarrow \infty$，所有类别概率趋近均匀分布 $1/K$