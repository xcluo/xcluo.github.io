### Sound Modality
1. 直接整体对应一个embedding，[seq_len, dim]
2. sound序列，[seq_len, K, dim]，一般K≤8
3. 第2维分别为声母、韵母、介母,即 [seq_len, 3, dim]，一般通过CNN学习得到一个句sound向量表示`flatter_embedding.shape=(seq_len, dim)`，
    - 当样本数够多时，可直接基于足量数据学到泛化功能，此时效果等同于方法1（sound整体对应一个embedding）
    - 当样本数不是很多时，可以考虑使用该方法，通过拆解元素学到泛化效果（两个字pinyin在某个编辑距离内相似度较高）
  
!!! info ""
    1. 上述方法均可指定是否带声调
    2. 提前获取pinyin映射表，多音字取最常用的那个以加快速度
    3. 字音sound需与token一一对应，sound缺失可直接用token替代，<span style="color:red;">每个table互相独立且最好采用相同dim，否则极容易发散</span>
    4. 最好只对汉字、数字和字母进行pinyin化（字母和单个字母pinyin需要进行区分）
