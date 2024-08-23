### declaration

#### get_variable
1. 初始化variable
```python
tf.get_variable(
    name,               # variable_name
    shape=None,         # variable_shape
    dtype=None,         # 数据类型
    initializer=None,   # 使用了具体数值进行初始化时不能指定shape
    trainable=None)
```

```
with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
    tf.get_variable()   # 未定义重新定义
                        # 定义了复用直接取值（类中定义也算）
```

#### Variable
每次调用都是创建新对象，且检测到命名冲突时，该函数会自动处理冲突(改名)并完成对象创建，因此即使 `reuse=True` 时该方法也无法实现共享变量，其余效果与 [get_variable](#get_variable)完全一致

#### variable_scope

#### name_scope



#### embedding
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


### initialization
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
`glorot` 在2010年由Xavier glorot发明，所以也叫做`Xavier`，用于对线性变换的张量(d_in, d_out)进行初始化  

- `uniform`，$\text{lim}=\sqrt{\frac{6}{d\_in + d\_out}}$，区间为 `[-lim, lim]`
- `normal`，$\text{stddev}=\sqrt{\frac{2}{d\_in + d\_out}}$，等价于`truncated_normal_initializer(mean=0, stddev=stddev)`


```python
def __init__(self, seed=None, dtype=dtypes.float32):
```