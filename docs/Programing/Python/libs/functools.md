### `iterable`操作
#### cmp_to_key
将比较函数转化为`key`函数
```python
# 设置比较函数
def my_cmp(a, b):
    ...
# 重写比较函数
my_list.sort(key=cmp_to_key(my_cmp))
```

#### reduce
将一个序列归纳为一个输出
```python
def reduce(
    function,       # 归纳函数
    sequence,       # 参与归纳的序列，即 `[y1, y2, y3, ...]`
    initial=None)   # 初始值，即 `x`，未指定时通过`sequence` 进行初始化 `x, *y = sequence`

alist = range(1, 50)
print(reduce(lambda x, y: x + y, alist))  # (1+50)*50/2 - 1225
```

### function 操作
#### partial
创建偏函数方法，即基于一个已有的函数（通过预设该函数的一些参数或关键字），生成一个新的函数
```python
def func(a, b, c):
    return a + b + c

# 使用partial冻结部分参数
new_func = partial(func, 1, 2)

# 调用新函数，只需要提供未冻结的参数
result = new_func(3)    # 等同于 func(1, 2, 3)
print(result)           # 输出6
```

#### partialmethod
与`partial`类似，但是它应用于方法而不是函数
```python
class Greeter:
    def __init__(self, greeting):
        self.greeting = greeting

    def greet(self, name, *args):
        return f"{self.greeting}, {name} {''.join(args)}"

    # 创建一个`partialmethod`的方法
    greet_hello = functools.partialmethod(greet, 'Hello')


# 创建一个 Greeter 实例
greeter = Greeter('Bonjour')

# 正确调用`partialmethod`的方法
print(greeter.greet_hello("Alice!"))  # 输出: Bonjour, HelloAlice!
```

### 装饰器
#### total_ordering
在定义类时只定义一小部分比较方法，然后它会自动补全其余的比较方法，这个装饰器要求

- 类中**至少定义了一个** `__lt__`、`__le__`、`__gt__` 或 `__ge__` 中的一个方法
- **必须定义** `__eq__` 方法

```python
@total_ordering
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

    def __eq__(self, other):
        return self.grade == other.grade

    def __lt__(self, other):
        return self.grade < other.grade
```

#### wraps

#### update_wrapper

#### singledispatch

#### singleDispatchMethod