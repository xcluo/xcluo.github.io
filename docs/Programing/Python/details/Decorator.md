
### 类内装饰器
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