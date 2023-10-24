### tqdm
```python
from tqdm import tqdm

class tqdm(Comparable):
    def __init__(self, 
            iterable=None,  # 待计数
            desc=None,      # 计数/进度条刷新时的描述文本
            total=None,     # 百分比进度条分母
            ...,
            **kwargs):
```
#### 1. `Iterator`和`Generator`：计数显示
无法直接获取长度`total`的`iterable`执行计数显示
```python
# 无法确定参数值total
for element in tqdm(iterable=iterator_var):  # 每次计数值 +1
for element in tqdm(iterable=generator_var): # 每次计数值 +1
```
#### 2. `Iterable`：百分比显示
可直接获取长度`total`的`iterable`执行进度百分比显示
```python
# 直接定义参数total
pbar = tqdm(total=total_num)
pbar.update(n)                               # 每次更新 n/total_num * 100%

# 通过len(iterable)定义参数total
for element in tqdm(iterable=iterable_var):  # 每次更新 1/len(iterable) * 100%
```

### trange
```python
from tqdm imprt trange

def trange(*args, **kwargs):
    """
    A shortcut for tqdm(xrange(*args), **kwargs).
    On Python3+ range is used instead of xrange.
    """
    return tqdm(_range(*args), **kwargs)
```
>- `assert len(args) == 1 or len(args) == 2 or len(args) == 3`  
>-  `trange` 只输入由`[left: right: step]`的序列，`tqdm(iterable)`的自定义性高
>- `range`返回的是一个`Iterable`，所以结果也是百分比进度条显示