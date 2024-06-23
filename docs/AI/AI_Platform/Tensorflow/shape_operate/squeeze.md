#### squeeze
```python
# 将张量 `input` 中的维度为1的轴（若指定了轴只处理对应的轴）删除
def squeeze(
    input,
    axis=None,          # Union[int, list[int]]
    name=None, 
    squeeze_dims=None   # `axis`
)
```
#### unsqueeze
通过 `tf.expand_dims` 来实现unsqueeze效果
```python
# 向张量 `input` 插入一个维度为1的轴
def expand_dims(
    input,
    axis=None,      # int, 和python类似，支持负数反向索引
    name=None,
    dim=None        # `axis`
)
```