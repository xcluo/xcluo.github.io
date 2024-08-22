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
    
    > 多个embedding拼接：界定每个embedding的vocab_size，借助 `where` 方法通过`emb_size_m <= v emb_size_n`批次获取后通过MLP以解决不同大小的dim拼接

【有道云笔记：索引与切片】split, gather, where
#### 张量声明
- 变量声明：get_variable
- 随机数声明：[initializer]()

#### 参数调度策略
- 学习率调度：[lr_schedule](schedule/lr_related.md)、[warm_up]()
    
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


```
with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
    tf.get_variable()   # 未定义重新定义
                        # 定义了复用直接取值（类中定义也算）
```