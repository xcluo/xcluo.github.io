train

    1. 多流输入input_stream_i
    2. 多流输出output_stream_i
    3. 融合多模态fusion
    4. loss = Σloss(output_stream_i) + loss(fusion)

infer

    1. 多流输入input_stream_i
    2. 多流输出output_stream_i
    3. 融合多模态fusion
    4. max(output_stream_i_prob, ..., fusion)

#### 字音sound模态
- 直接整体对应一个embedding，[seq_len, dim]
- sound序列，[seq_len, K, dim]，一般K≤8
- 区分声母、韵母、介母, [seq_len, 3, dim]，一般通过CNN学习得到一个flatter_embedding，
    - 当样本数够多时，可直接基于足量数据学到泛化功能，此时效果等同于方法1（sound整体对应一个embedding）
    - 当样本数不是很多时，可以考虑使用该方法，通过拆解元素学到泛化效果（两个字pinyin在某个编辑距离内相似度较高）
  
!!! info ""
    1. 上述方法均可指定是否带声调
    2. 提前获取pinyin映射表，多音字取最常用的那个以加快速度
    3. 字音sound需与token一一对应，sound缺失可直接用token替代（文本中可能存在用pinyin替代token的变种，因此token和sound可使用同一张embedding表）
    
#### 字形shape模态
- 笔画拆解
- 上下结构，左右结构
!!! info ""
    1. 字形shape需与token一一对应，shape缺失可直接用缺省状态（如UNK）替代（shape可独立为一张embedding表）
   