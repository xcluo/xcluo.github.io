

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