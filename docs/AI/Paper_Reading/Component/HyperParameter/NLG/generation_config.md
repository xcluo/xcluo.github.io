#### do_sample
控制模型是否从基于概率的候选词中随机采样获取下一个词还是确定性方法生成下一个词。

#### num_beams/beam_width
使用beam search束搜索方式生成序列**（只要seed一致，生成的序列集合是确定性的）**，维护一个固定大小的候选序列集合（概率最高的k个序列），并逐步扩展这些序列直到达到预定的最大长度或满足其他停止条件。

#### length_penalty
调节生成序列的长度偏好

#### early_stopping
当 `early_stopping=True` 时，束搜索会在所有当前的候选序列都达到了结束条件（例如遇到了结束标记 `<eos>` 或者达到了最大长度 `max_length`）时提前停止搜索。这意味着即使没有达到最大步数，只要所有的候选序列都完成了，搜索就会停止。
#### repetation_penalty

#### top-k
对top-k个候选词进行采样作为下一生成词。

#### top-p
即nuclear sampling，对按置信度降序排列的前n个词进行采样作为下一生成词，要求满足$\sum_{i=1}^{n-1} p_i \lt p\text{ and }\sum_{i=1}^{n} p_i \ge p$
