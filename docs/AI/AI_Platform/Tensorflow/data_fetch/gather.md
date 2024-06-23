`tf.gather` 基于给定的索引坐标 `indices` 从的张量 `param` 的`axis` 维度获取 ^^切片并整合^^ 后的张量
```python
def gather(
    param,
    indices,                # isinstance(indices, ndaddry or list or tensor)
                            # ndims(indices) in {0, 1}
    validate_indices=None,  # pass形参，无效用
    name=None,
    axis=None,              # 指定切片的轴
    batch_dims=0
)
```