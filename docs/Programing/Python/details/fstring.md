

`" %s %d"`

#### `str.format()`
使用 {} 和 : 来替代以前的 %

1. 传参
```python
# 不设置指定位置，按默认顺序传参
"{} {}".format("hello", "world")            # 'hello world'

# 设置指定位置传参
"{1} {0}".format("hello", "world")          # 'world hello'

# 关键字传参                                    
"{v1} {v0}".format(v0="hello", v1="world")  # 'world hello'

# 索引传参
vars = ['菜鸟教程', 'www.runoob.com']
"{0[1]} {0[0]}".format(vars)                # 'world hello'
```
1. 指定格式 & 自动补全 & 位数保留
```python
# 整型
{:c/</>/^/nd}
                    # c：填充字符，缺省为空格
                    # 方向，{<: 右侧填充; >: 左侧填充; ^: 左右(奇数时右侧多)填充，中间对齐}
                    # n：最小位数
                    # d：指定为整型

# 浮点型
{.nf}               # n：小数保留位数
                    # f：指定为浮点型

# 其余进制
{:b}                # 无前缀0b的2进制形式
{:#b}               # 有前缀0b的2进制形式
{:o}                # 无前缀0o的8进制形式
{:#o}               # 有前缀0o的8进制形式
{:x}                # 无前缀0x的16进制 + 字母小写 形式
{:#x}               # 有前缀0x的16进制 + 字母小写 形式
{:X}                # 无前缀0x的16进制 + 字母大写 形式
{:#X}               # 有前缀0x的16进制 + 字母大写 形式
```
!!! info ""
    在保留小数点后n位时，获取保留小数的位数 `f'{val:.{n}f}'`
    ```python
    import numpy as np
    def get_str_length(val):
        val_str = np.format_float_positional(val, trim='-')
        n = len(val_str.split('.')[-1])
        return n
    ```
#### fstring
等价于 [str.format()](#strformat) 的关键字传参方式，区别在于 字符串左侧使用 `f""`修饰且不需要后面的 `.format()`
