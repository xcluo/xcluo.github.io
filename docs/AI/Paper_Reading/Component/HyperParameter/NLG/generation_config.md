#### do_sample
控制模型是否从基于概率的候选词中随机采样获取下一个词还是确定性方法生成下一个词。

#### num_beams/beam_width
使用beam search束搜索方式生成序列**（只要seed一致，生成的序列集合是确定性的）**，维护一个固定大小的候选序列集合（概率最高的k个序列），并逐步扩展这些序列直到达到预定的最大长度或满足其他停止条件。

#### early_stopping
当 `early_stopping=True` 时，束搜索会在**所有当前的候选序列**都达到了结束条件（例如遇到了结束标记 `<eos>` 或者达到了最大长度 `max_length`）时提前停止搜索。这意味着即使没有达到最大步数，只要所有的候选序列都完成了，搜索就会停止。

#### temperature
温度系数，$p_i=\frac{e^{z_i/T}}{\sum_j e^{e_j/T}}$，温度值越高（差异变小），概率分布越平滑，生成文本越多样化；反之概率分布越尖锐，生成文本更趋近高概率结果。

#### length_penalty
该参数通过对每个候选序列的概率进行调整，以平衡较短和较长序列之间的概率，最终调节生成序列的长度偏好，具体过程如下

$$
P^{'}=\frac{P}{(L+1)^\alpha}
$$

!!! info ""
    - $P$ 表示原始**序列积累概率**($-\log\big(\prod p_1\cdots p_{L}\big)$)，$L$ 表示序列长度，$\alpha$ 为`length_penalty`
    - `α > 0`，幂底数<1，因此倾向于生成更短的序列(由长度引起的序列积累概率影响没幂指数函数大)
    - `α < 0`，幂底数>1，因此倾向于生成更长的序列(由长度引起的序列积累概率影响没幂指数函数大)
    - `α = 0`，无长度惩罚
    - `α = 1`，缺省状态，鼓励生成更短的序列

#### repetation_penalty
该参数用于控制模型生成文本时对重复词语或短语的惩罚程度(调整词概率)，具体过程为：  

1. 调整候选词概率，$R$表示`repetation_penalty`  

    $$
    p^{'}(x)=\begin{cases}
        {p(x)^{1/R}} & x\text{ has occurred} \\
        {p(x)} & x\text{ has not occurred}
    \end{cases}
    $$

2. 归一化候选词概率  

    $$
    p^{''}(x)=\frac{p^{'}(x)}{\sum_y p^{'}(y)}
    $$

!!! info ""
    - 实际应用中`R`通常设定在0.8到1.2之间
    - `R > 1.0`，增加模型对重复词的惩罚，降低出现概率，鼓励多样性  
    - `R < 1.0`，降低模型对重复词的惩罚，提升出现概率，可能有助于保持上下文一致性  
    - `R = 1.0`，缺省情况，不对该重复单词进行任何特别处理

#### top-k
对top-k个候选词进行采样作为下一生成词。

#### top-p
即nuclear sampling，对按置信度降序排列的前n个词进行采样作为下一生成词，要求满足$\sum_{i=1}^{n-1} p_i \lt p\text{ and }\sum_{i=1}^{n} p_i \ge p$