Tensorflow的功能是定义一个计算图，通过`sess.run()`来启动计算图得到输出结果。
!!! info ""
    计算接过前需要先提前对计算图参数进行初始化操作
    ```python
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    ```

#### 张量属性
shape.as_list()  
shape.ndims  
shape.dims  

#### 数据获取
- 训练数据生成：[Dataset](data_fetch/Dataset.md)
- 切片：[gather](data_fetch/gather.md), [where](data_fetch/where.md)
- 型操作：[squeeze](shape_operate/squeeze/#squeeze), [expand_dims](shape_operate/squeeze/#unsqueeze)；[split](shape_operate/split_concat/#split), [concat](shape_operate/split_concat/#concat)

【有道云笔记：索引与切片】split, gather, where

#### 张量声明
- 变量声明：[get_variable](tensor_related/declaration)、[Variable]()
- embedding相关
- 随机初始化：[initializer]()

#### 参数调度策略
- 梯度更新：[gradient_update](schedule/gradient_update.md)、[gradient clipping](schedule/gradient_update/#gradient-clipping)
- 学习率调度：[lr_schedule](schedule/gradient_update/#lr_schedule)、[warm_up](schedule/gradient_update/#warmup)
- 优化器：[optimizer](schedule/gradient_update/#optimizer)
- 扩散模型β调度：[beta_schedule]()




