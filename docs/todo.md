- MinHash de-duplication
- Frobenius范数，次可加性$\Vert A+B \Vert_{F}\le \Vert A \Vert_F + \Vert B \Vert_F$，空间向量相加，两边之和大于第三边
- LDA潜在迪利克雷分布，b站视频 LDA主题模型
- LSA/PLSA
- Attention softmax后除以$\sqrt{d_k}$是因为权重矩阵中每个元素都是通过两个(1， d_k)方差为1的向量相乘得到的，基于正态分布累加后的标准差公式可知该值方差变为$\sqrt{d_k}$，因此执行该操作
- tensorflow 1.x中在梯度下降时如何设置L1，L2正则化约束
    ```python
    with tf.variable_scope("layer1", regularizer=l2_regularizer):
        xxx
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = cross_entropy_loss + sum(reg_losses)
    ```