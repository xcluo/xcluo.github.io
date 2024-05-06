Tensorflow的功能是定义并初始化一个计算图，通过`sess.run()`来执行这个计算图。

#### 数据获取
- 训练数据生成：[Dataset](data_fetch/Dataset.md)
- 指定张量获取：[gather](), [where]()
- 张量划分、联结：[split]()，[reshape]()
【有道云笔记：索引与切片】split, gather, where
#### 张量声明
- 变量声明：get_variable
- 随机数声明：[initializer]()

#### 参数调度策略
- 学习率调度：[learning_rate_schedule]()、[warm_up]()
    ```python
    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=global_step)

    sess.run(train_op)
    ```
- 扩散模型β调度：[beta_schedule]()

1. 初始化variable
```python
tf.get_variable(
    name,
    shape=None,             # 使用常数（如ndarray）时初始化不需要指定shape
    dtype=None,             # 
    initializer=None        # 使用常数（如ndarray）时初始化不需要指定shape
)
```

1. 在矩阵embedding中提取一组索引对应的子集
```python
tf.nn.embedding_lookup(
    params,
    ids,
    partition_strategy="mod"
    name=None    
)
```
!!! info
    功能等同于gather，该方法高效实用，通过将较大的params分块存储，再根据ids和每块的位置进行索引取值 https://www.zhihu.com/question/52250059

