### 数值相关
1. `min/max`：获取`iterable`中最小值/最大值
```python
min/max(iterable, *, key=None)
min/max(*args, key=None)
```
> 如果多个元素符合`key`对应的条件，返回第一次出现的元素
1. `sum`：返回`iterable`元素值的和，即
```python
ret = 0
for e in iterable:
    ret += e
return ret
```
1. `pow`：执行幂指数函数，如果指定求模函数底，进一步求模
```python
pow(
    base,       # 指数函数底
    exp,        # 指数函数幂指数
    mod=None    # 求模函数底，未指定不求模
)
```

1. `round`

### 命令执行相关
1. `eval/exec`
```python
exec/eval(
    expression,     # type(expression)=str，表示要执行的语句
    globals=None,
    locals=None,
    /
)
# eval执行返回结果，exec执行不返回结果
a = 1
print(eval('a+2'), a)   # > None, 3
print(exec('a+2'), a)   # > 3, 1
```
> `eval`函数的`expression`参数出现`=`时会报错`SyntaxError`  
> 出现修改传入值的情况下一般会选择使用`exec`函数


1. `callable`：判断对象是否是可调用的
```python
callable(
    object
) -> bool
```
> 当`object`是一个方法、函数或者类时是可调用的；当`object`是个类对象且类中实现了 `__call__` 方法时也是可调用的

### 类型相关
1. `chr/ord`：将单个整型数值转换为对应的unicode字符串/将单个unicode字符转化为整型数值
```python
# chr
chr(
    i                   # type(i)=int, 表示要转换的整型数值，0 <= i <= 0x10ffff
) -> str

# ord
ord(
    c                   # type(c)=str, 表示要转换的unicode字符
) -> int
```
> `ord` 与 `chr` 功能相反
1. `str`
1. `int`
1. `float`
1. `bool`
1. `bin`
1. `hex`
1. `oct`
1. `type`
1. `isinstance`

### 容器相关
1. `len`
1. `list`
1. `tuple`
1. `dict`
1. `set`
1. `zip`
1. `range`
1. `enumerate`
1. `filter`
1. `map`
1. `iter`
1. `next`
1. `sorted`
1. `reversed`
1. `all/any`

判断 `Iterable` 中元素状态，当元素为以下值之一，元素逻辑状态值为`False`，否则为`True`

- `None`
- `0`
- `Iterable`中元素个数为0
- `False`
> `iterable` 中各元素可为条件逻辑，通过`all`或`any`以组建为`or`或`and`逻辑表达式