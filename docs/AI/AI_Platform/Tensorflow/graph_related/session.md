#### tf.Session
主要用于执行网络。所有关于神经网络的计算都在这里进行，它执行的依据是计算图或者计算图的一部分，同时，会话也会负责分配计算资源和变量存放，以及维护执行过程中的变量。
```python
def __init__(self,
    target='',
    graph=None,             # tf.Graph，
    config=None):           # tf.ConfigProto，设置session的各种配置选项
```


#### tf.ConfigProto
https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto
```python
# 直接初始化
def session_config = tf.ConfigProto(
        log_device_placement=True,          # 日志打印出TensoFlow使用了哪种操作
        allow_soft_placement=True,          # 根据设备可用情况，自动分配GPU或CPU
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0)
# 单独赋值
cofnig.log_device_placement=True
config.allow_soft_placement=True
config.inter_op_parallelism_threads=1
config.intra_op_parallelism_threads=1
config.gpu_options.per_process_gpu_memory_fraction=0.1      # 指定最大的gpu使用百分比
config.gpu_options.allow_growth=True                        # 动态分配gpu，用多少取多少
```
> 不指定`gpu_options`时默认使用全部gpu
#### Session.run & Tensor.eval