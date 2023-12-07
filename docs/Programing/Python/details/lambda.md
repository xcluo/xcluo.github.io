```python
lambda *x: sum(x)
```

```python
import functools

# 由于callable的输入是一个函数，且不存在参数列表，所以可借助functools.partial实现传参
new_fun_name = functools.partial(sum, x)
```