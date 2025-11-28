Tensorflow的功能是定义一个计算图Graph，通过`sess.run()`来启动计算图得到结果。
!!! info ""
    计算接过前需要先提前对计算图参数进行初始化操作
    ```python
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    ```

#### 图相关
- 计算图：[Graph](graph_related/ops/graph.md)
    - 边：[Tensor](graph_related/ops/tensor.md)
    - 节点：[Operation](graph_related/ops/operation.md)
- 会话：[Session](graph_related/session.md)


#### 数据获取
- 训练数据生成：[Dataset](data_fetch/Dataset.md)
- 切片：[gather](data_fetch/gather.md), [where](data_fetch/where.md)
- 型操作：[squeeze](shape_operate/squeeze/#squeeze), [expand_dims](shape_operate/squeeze/#unsqueeze)；[split](shape_operate/split_concat/#split), [concat](shape_operate/split_concat/#concat)

【有道云笔记：索引与切片】split, gather, where

`tf.einsum` 基于爱因斯坦在相对论研究中引入的简化标记法。其基本规则是：如果在一个乘积项中，某个下标出现了两次，那么就自动对这个下标进行求和。

- 向量点积为 $c \sum_i a_i b_i$，由于下标i出现了两次，因此可以表示为 `tf.einsum("i,i->", a, b)`
- 向量外积 `tf.einsum("i,j->ij", a, b)`
- 矩阵乘法 `tf.einsum("ij,jk->ik", a, b)`
- 高阶张量运算 `tf.einsum("ijk,kl->ijl", a, b)`
- 矩阵逐元素相乘 `tf.einsum("ij,ij->ij", a, b)`
> 直接通过指定型的变换进行运算



#### 张量声明
- 变量声明：[placeholder](tensor_related/declaration/#placeholder)、[get_variable](tensor_related/declaration/#get_variable)、[Variable](tensor_related/declaration/#variable)
- embedding相关：[embedding](tensor_related/declaration/#embedding)
- 初始化：[initializer](tensor_related/declaration/#initializer)


#### 参数调度策略
- 梯度更新：[gradient_update](schedule/gradient_update.md)、[gradient clipping](schedule/gradient_update/#gradient-clipping)
- 学习率调度：[lr_schedule](schedule/gradient_update/#lr_schedule)、[warm_up](schedule/gradient_update/#warmup)
- 优化器：[optimizer](schedule/gradient_update/#optimizer)
- 扩散模型β调度：[beta_schedule]()

#### 网络层相关

