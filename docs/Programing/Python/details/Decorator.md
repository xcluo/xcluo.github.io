---
title: "decorator"
---

### 函数装饰器
函数装饰器是一个函数，它接受被装饰函数作为参数，在不改变该被装饰函数的前提下为其增加新的功能

#### 基本装饰器
=== "装饰无参函数"
    ```python
    def my_decorator(func):
        def wrapper():
            print("函数执行前")
            func()
            print("函数执行后")
        return wrapper

    # say_hello作为函数my_decorator的参数
    @my_decorator
    def say_hello():
        print("Hello!")

    say_hello()

    # 函数执行前
    # Hello!
    # 函数执行后
    ```

=== "装饰带参函数"
    ```python
    def my_decorator(func):
        def wrapper(*args, **kwargs):
            print(f"调用函数: {func.__name__}")
            result = func(*args, **kwargs)
            print("函数执行完成")
            return result
        return wrapper

    # greet作为函数my_decorator的参数
    @my_decorator
    def greet(name):
        print(f"Hello, {name}!")

    greet("Alice")

    # 调用函数: greet
    # Hello, Alice!
    # 函数执行完成
    ```

#### 带参装饰器
```python
def repeat(num_times):
    def decorator_repeat(func):
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator_repeat

@repeat(num_times=3)
def say_hello():
    print("Hello!")

say_hello()

# Hello!
# Hello!
# Hello!
```
#### 装饰器嵌套
顺序执行，类似于递归调用
```python
def decorator1(func):
    def wrapper():
        print("装饰器1 - 前")
        func()
        print("装饰器1 - 后")
    return wrapper

def decorator2(func):
    def wrapper():
        print("装饰器2 - 前")
        func()
        print("装饰器2 - 后")
    return wrapper

@decorator1
@decorator2
def say_hello():
    print("Hello!")

say_hello()

# 装饰器1 - 前
# 装饰器2 - 前
# Hello!
# 装饰器2 - 后
# 装饰器1 - 后
```

### 类装饰器

### 方法装饰器
#### @staticmethod
可以单独摘出类中，只是为了便于分类目的放入到类中，本质还是和类独立（不用用到类对象、对象属性以及类属性）

#### @classmethod
只能访问类属性，不能访问对象属性

#### @property
在方法定义上面加一个 `@property` 装饰器，就可以把方法变成一个属性，即通过 `object_name.property_name` 来访问
```python
class Person:
    ...
    @property
    def fullname(self):
        return self.first + ' ' + self.last

```

#### @<property_name\>.setter
对于`@property`修饰的对象属性的赋值，可以用`@<property_name>.setter`修饰的方法来实现
```python
class Person:
    ...
    @fullname.setter
    def fullname(self, name):
        first, last = name.split(' ', 1)
        self.first = first
        self.last = last
```

####  @wraps
![alt text](9147dba1047a2774a36f509260a0c5e7_61834.png)
```python
say_hello.__name__  # my_decorator（在修饰器内输出func.__name__才为say_hello）
say_hello.__doc__   # my_decorator的函数文档（在修饰器内输出func.__doc__才为say_hello的函数文档）
```