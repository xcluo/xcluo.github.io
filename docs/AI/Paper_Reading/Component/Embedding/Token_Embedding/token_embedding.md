### Token Modality

#### 多段embedding_table拼接
通过连接多个不同来源的embedding_table，以增加词表vocab的多样性（如汉字、字符、emoji等）；又由于多种不同来源的embedding_table的特征数不一致，因此需要：

1. 选取各embedding_table中对应的向量
2. 投影至同一维度`dim`
3. 拼接成一个最终的`sequence embedding`
=== "TF 1.x"
    ```python
    word_embedding = 0.
    for i, offset in enumerate(offsets):
        word_embedding += tf.nn.embedding_lookup(
            tf.get_variable(f"word_embeddings_part_{i}"),
            word_ids - offset
        )
    ```
!!! info ""
    `tf.nn.embedding_lookup`：当`id ∈ [0, vocab_size)` 时选取对应的embedding，否则返回`[0, 0, ..., 0]`
### Sound Modality

### Shape Modality