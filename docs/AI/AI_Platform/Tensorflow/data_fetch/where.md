`tf.where` 具有以下两种效果：

1. ^^如果`x`和`y`都为None^^，则该操作将返回`condition`中所有true元素的索引idx
2. ^^如果x和y都不为None^^，则要求 `shape(x) == shape(y)` 且 `shape(x) == shape(condition) or shape(x)[0] = shape(condition)`
```python
def where(
    condition,      # condition.dtype == tf.bool
    x=None,
    y=None,
    name=None
)
```