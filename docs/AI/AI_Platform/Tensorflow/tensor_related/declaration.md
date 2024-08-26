### declaration

#### placeholder
为需要输入的张量插入一个占位符，对应`sess.run`中的`feed_dict`
```python
def placeholder(
    dtype,
    shape=None,                 
    name=None)
```

#### get_variable
```python
tf.get_variable(
    name,                       # variable_name
    shape=None,                 # variable_shape
    dtype=None,                 # data type
    initializer=None,           # 使用了具体数值进行初始化时不能指定shape
    trainable=None)
```

1. 搭配`variable_scope`实现变量共享
```python
with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
    tf.get_variable(var_name)   # true_var_name=f'{scope_name}/{var_name}'
                                # scope_name="", var_name=true_var_name等价于↑
    # reuse: 1/True/tf.AUTO_REUSE，未定义则创建，已定义则获取该变量
    # reuse: 0/False/None，未定义则创建，已定义报错`ValueError`并提示变量已存在
```
2. 使用[`initializer`](#initializer)初始化
```python
tf.get_variable(
    name,
    shape=[dim_1, ..., dim_n],
    initializer=tf.truncated_normal_initializer(mean, stddev),
    dtype=tf.float32
)
```
3. 指定数值初始化
```python
tf.get_variable(
    name,
    initializer=pkl.load(open(value_file_path, 'rb'))
    dtype=tf.float32
)
```
> 使用指定值初始化时不能指定`shape`
#### Variable
每次调用都是创建新对象，且检测到命名冲突时，该函数会自动处理冲突(改名)并完成对象创建，因此即使 `reuse=True` 时该方法也无法实现共享变量，其余效果与 [get_variable](#get_variable) 完全一致 
> <span style="color: red;">使用`tf.Variable`</span>声明，而后用`tf.get_variable`获取对应名字的变量也<span style="color: red;">无法实现变量共享</span>。



#### embedding
embedding一般为一个`[vocab_size, dim]` 的二维矩阵张量

1. 在矩阵embedding中提取一组索引对应的子集
```python
# params([vocab_size, dim]) + ids([seq_len]) → representation([seq_len, dim])
tf.nn.embedding_lookup(
    params,
    ids,
    partition_strategy="mod"
    name=None    
)
```
!!! info
    功能等同于[gather](../data_fetch/gather.md)，但该方法更为高效，通过将规模较大的params分块存储，再根据ids和每块的位置进行索引实现取值 https://www.zhihu.com/question/52250059


### initializer
#### zeros/ones/constant_initializer
全使用 `0/1/value` 进行全部初始化
```python
''' tf.zeros/ones_initializer(dtype=tf.float32) '''
def __init__(self, dtype=dtypes.float32):
''' tf.constant_initializer(2, dtype=tf.float32) '''
def __init__(self, value=0, dtype=dtypes.float32, verify_shape=False):
```

#### random_uniform_initializer
每个值都从 `[min_val, max_val]` 区间内均匀采样进行初始化
```python
def __init__(self, minval=0, maxval=None, seed=None, dtype=dtypes.float32):
# if maxval == None; maxval=1
# minval, maxval = min(minval, maxval), max(minval, maxval)
```
#### random/truncated_normal_initializer 
- `random`：每个值都从分布 $N~(\mu, \sigma^2)$ 中采样进行初始化
- `truncated`：将保留采样中处于 `[μ-2*σ, u+2*σ]` 区间内的值，未处于区间内的值重新进行采样直至处于该目标区间（一般`stddev=0.02`）

```python
def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=dtypes.float32):
```
#### glorot_uniform/normal_initializer
`glorot` 在2010年由Xavier glorot发明，所以也叫做`Xavier`，用于对线性变换层的张量(d_in, d_out)初始化  

- `uniform`，$\text{lim}=\sqrt{\frac{6}{d\_in + d\_out}}$，区间为 `[-lim, lim]`
- `normal`，$\text{stddev}=\sqrt{\frac{2}{d\_in + d\_out}}$，等价于`truncated_normal_initializer(mean=0, stddev=stddev)`


```python
def __init__(self, seed=None, dtype=dtypes.float32):
```