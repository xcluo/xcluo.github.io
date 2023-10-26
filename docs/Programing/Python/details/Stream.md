### 返回`Iterable`
#### 1. `filter`
#### 2. `map`
#### 3. `zip`
#### 4. `sorted`
#### 5. `reversed`

### 返回`bool`值
判断 `Iterable` 中元素状态，当元素为以下值之一，元素逻辑状态值为`False`，否则为`True`

- `None`
- `0`
- `Iterable`中元素个数为0
- `False`
> `iterable` 中各元素可为条件逻辑，通过`all`或`any`以组建为`or`或`and`逻辑表达式
#### 1. `any`
是否存在某一元素逻辑状态值为`True`
```python
any(iterable)
```
#### 2. `all`
是否所有元素逻辑状态值为`True`
```python
all(iterable)
```