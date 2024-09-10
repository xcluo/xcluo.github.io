tensorflow/python/frameworkd/ops

### tf.Graph
由节点和有向边描述数学运算的有向无环计算图，主要用于构建网络（设计启发是高等数学里面的链式求导法则的图），本身不进行任何实际的计算，可以将计算图理解为是一个计算模板或者计划书。

#### graph
1. `graph.as_default()`  
将当前graph设为缺省graph

2. `tf.get_default_graph()`  
类似于`get_variable`，未创建便创建图并设为缺省graph，创建了就直接获取缺省graph

#### _collections
1. `graph._collections: defaultdict(list)`  
以集合的形式存放图中的变量，每次创建变量时调用`tf.add_to_collection(s)`进行存储

1. `graph.collections`  
以list形式返回`_collections`的key


1. `tf.GraphKeys`  
枚举类，包含`_collections`中常见的key
```python
LOCAL_VARIABLES = "local_variables"             # local variables，声明时指定`collections=[tf.GraphKeys.LOCAL_VARIABLES]` 的变量
GLOBAL_VARIABLES = "variables"                  # global variables
TRAINABLE_VARIABLES = "trainable_variables"     # trainable variables
```


1. `tf.trainable_variables()`  
获取trainable的变量
```python
# 等价的表述方式
tf.trainable_variables(scope=None)
tf.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope=None)
```

节点Operation: name + op + input + attrs  
边Tensor：

session=run operation + eval tensor