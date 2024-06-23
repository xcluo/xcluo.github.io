#### split
两种用法：

1. `isinstance(num_or_size_splits, list)`：指定每部子张量在`axis`轴上划分的数量
2. `isinstance(num_or_size_splits, int)`：每部分子张量在`axis`轴上等量划分，不足的情况下只取剩余的部分
```python
# 沿指定轴`axis`将一个张量`value`划分为多个张量
def split(
    value,
    num_or_size_splits,     # Union[int, list[int]]
    axis=0,                 # int, 和python类似，支持负数反向索引
    num=None,
    name="split"
)
```

#### concat
```python
# 沿指定轴`axis`将多个张量`values`整合为一个张量
def concat(
    values,                 # list[tf.Tensor]
    axis,                   # int, 和python类似，支持负数反向索引
    name="concat"
)
```