Python中 `file.write()` 方法<span class='hl_span'>只能写入字符串</span>，调用该方法写入数据时都要转化为 `str` 类型。
> `file` 为 `open()` 方法返回的文件对象

### bin
```python
# 读取bin文件
with open(file_name, mode='rb') as file:    # mode=’rb' 控制以字节方式读取，bin文件不设置encoding
    line = f.read().decode(encoding)        # 通过decode方法将读取的bytes转化为字符串形式，便于处理
    
# 写入bin文件
with open(file_name, mode='wb') as file:    # mode=’wb' 控制以字节方式写入，bin文件不设置encoding
    f.write(string.encode(encoding))        # 通过encode方法将待写入bytes转化为字符串形式 (文件只能写入字符串)
```

### json

```python
import json

# 字符串读写，字符串  <-转化-> json字典
def json.loads(
    s,
    encoding=None,                 # 编码方式
)

def json.dumps(
    obj,
    ensure_ascii=True,             # 是否将json字符串转化为ascii编码，为了可视化一般不转化
    sort_keys=Fals,                # 输出字典前是否对键 key 进行排序排序
    allow_nan=True,                # 
)


# 文件读写，文件 <-转化-> json字典
def json.load()
def json.dump()
```

>- 当直接写入或读取的文件过大时，速度很慢，json文件读写方法效率低
>- 且文件内容一般单行为一个json字典，一般逐行读取并消耗，较少考虑json文件读写方法

### pkl

```python
import pickle as pkl

# 文件读写
def pkl.load(
    file,                          # `open`方法返沪的文件对象
    fix_imports=True,
    encoding="ASCII",
    errors="strict",
)

# 文件写入
def pkl.dump(
    obj,                           # 待写入的对象，可以是任何可序列化的对象，无需手动转化为`str`类型
    file,                          # `open`方法返沪的文件对象
    protocol=None,
    fix_imports=True,
)
```