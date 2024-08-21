

```python
def fun_1(self, arg, *args):

# obj.fun_1(...) 调用时，方法自动赋值self=obj
# Class_Name.fun_1(obj, ...)，调用时，手动指定self=obj
# 二者等价

#@claclassmethod 调用
# Class_name.class_method
# obj.class_method
# 二者等价，因为类属性和不随对象变化而比变化

#成员属性只有通过类访问才可修改，即 Class_name.val = 3
```

### 变量

#### 类变量

#### 实例变量

### 方法

#### 类方法

#### 实例方法

### 继承

### 魔法属性

`__dict__`、`__slots__`、`__weakref__`、`__class__`

### 魔法方法

#### 对象相关
1. `__new__`，创建类对象，为静态方法，^^需要返回创建的对象^^，若未重写自动调用父类该方法
    ```python
    def __new__(cls, *args, **kwargs):  # *args和**kwargs等同于`__init__`对应的形参
        x = super().__new__(cls)
        # x = super(Clazz, cls).__new__(cls) # 基于静态方法特性，等同于上述写法
        return x                        # 返回的 `x` 即为其余方法的形参 `self` 
    ```
2. `__init__`，初始化创建的对象
    ```python
    def __init__(self, *args, **kwargs)
    ```


5. `__del__`，析构函数
    ```python
    def __del__(self):
        # 将对象从内存中销毁
        '''                   
        # 此时无法调用open函数写出数据，如需完成结果写出，可通过一下方法
        import atexit

        # in __init__
        atexit.register(<exit_function_name>)   # 在对象结束前完成结果写出
        '''
    ```

3. `__enter__`，与 `__exit__` 搭配，enable被 `with` 语句调用
    ```python
    def __enter__(self):
        self.file = open(self.file_name, self.mode)
        return self.file
    ```
4. `__exit__`，与 `__enter__` 搭配，enable被 `with` 语句调用
    ```python
    # 完成在`__enter__`方法中对应内容的关闭
    def __exit__(self, 
                 exc_type,      # exit_type 
                 exc_val,       # exit_vaule
                 exc_tb):       # exit_traceback
        self.file.close()
    ```
5. `__str__`，使用 `str`、`print` 以及 `format` 函数时调用的方法
6. `__repr__`，使用 `repr` 函数时调用的方法
    ```python
    # 调用优先级为 `__str__ -> __repr__`
    def __str__(self) -> str:
    # 调用优先级为 `__repr__`
    def __repr__(self) -> str:
    ```
7. `__call__`，直接使用 `object_name()` 时调用
    ```python         
    def __call__(self, *args, **kwargs)
    ```

#### 变量相关

```python
def __getattribute__(self, attr_name)
def __setattribute__(self, attr_name, value)

def __getattr__(self, attr_name)
def __setattr__(self, attr_name, value)
def __delattr__(self, attr_name)

def __get__(self, instance, owner)
def __set__(sefl, instance, value)
def __delete__(sefl, instance)
```

```python
__bool__
__int__
__float__
__format__
```


#### 容器相关

1. `__iter__`，使对象具有可迭代化属性（需同时搭配实现`__next__`方法）
```python
def __iter__(self) -> Iterator[T_co]:
    return self
```
> 只有实现了 `__iter__` 方法的类才能成功执行 `iter` 函数

1. `__next__`，获取序列下一个元素，（需同时搭配实现`__iter__`方法）
```python
def __next__(self) -> T_co
```
> 序列化的时候会遍历待序列化的对象，因此只有实现了同时实现了 `__iter__`和 `__next__` 方法的类才能成功序列化  
> 只有实现了 `__next__` 方法的类才能成功执行 `next` 函数

1. `__getitem__`，通过索引或键获取相应元素
```python
# 通过索引获取元素，如list、tuple、str等
def __getitem__(self, index) -> T_co

# 通过键获取值元素，如dict等
def __getitem__(self, key) -> T_co
```
> 只有实现了 `__getitem__` 方法的类才能成功执行 `[]` 运算符  
> 若类中未实现 `__iter__`、`__next__` 方法，Python解释器会找 `__getitem__` 来迭代

1. `__setitem__`，通过索引或键设置相应元素
```python
# 设置索引对应的元素值
def __setitem__(self, index, value)

# 设置键对应的元素值
def __setitem__(self, key, value)
```
> 只有实现了 `__setitem__` 方法的类才能成功执行 `[]=` 运算符

1. `__len__`，返回对象中元素个数
```python
def __len__(self) -> int
```
> 只有实现了 `__len__` 方法的类才能成功执行 `len` 函数

1. `__contains__`，判断对象中是否包含某个元素
```python
def __contains__(self, item) -> bool
```
> 只有实现了 `__contains__` 方法的类才能成功执行 `in` 运算符


#### 操作符重载

1. `__le/lt/eq/ne/gt/ge__`：数值比较
```python
def __le/lt/eq/ne/gt/ge__(self, other) -> bool
```
> 分别重载了二元操作符 `<=`、`<`、`==`、`!=`、`>`、`>=`

1. `__add/sub_mul/div/mod__`：数值运算
```python
def __add/sub_mul/div/mod__(self, other) -> T_co
```
> 分别重载了二元操作符 `+`、`-`、`*`、`/`、`%`
1. `__or/and/xor__`：按位运算
```python
def __or/and/xor__(self, other) -> T_co
```
> 分别重载了按位操作符 `|`、`&`、`^`
1. `__lshift/rshift__`：移位运算
```python
def __lshift/rshift__(self, other) -> T_co
```
> 分别重载了移位操作符 `<<`、`>>`

