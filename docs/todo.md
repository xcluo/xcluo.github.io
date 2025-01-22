- MinHash de-duplication
- Frobenius范数，次可加性$\Vert A+B \Vert_{F}\le \Vert A \Vert_F + \Vert B \Vert_F$，空间向量相加，两边之和大于第三边
- [x] LDA潜在迪利克雷分布，b站视频 LDA主题模型
- LSA/PLSA
- [ ] TF-IDF_j, MI_{a, b, c, d}
- Attention softmax后除以$\sqrt{d_k}$是因为权重矩阵中每个元素都是通过两个(1， d_k)方差为1的向量相乘得到的，基于正态分布累加后的标准差公式可知该值方差变为$\sqrt{d_k}$，因此执行该操作
- tensorflow 1.x中在梯度下降时如何设置L1，L2正则化约束
    ```python
    with tf.variable_scope("layer1", regularizer=l2_regularizer):
        xxx
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = cross_entropy_loss + sum(reg_losses)
    ```
- 推荐系统攻击
- [推荐系统论文笔记](https://github.com/Doragd/Algorithm-Practice-in-Industry/blob/main/%E6%90%9C%E5%B9%BF%E6%8E%A8%E7%AE%97%E6%B3%95%E7%B3%BB%E5%88%97%E4%B8%B2%E8%AE%B2.md#%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0)