#### 反斜杠转义问题
```python
import codecs
cnt = "仨竺汃昗宍竺訵宍泗\\n藤训号\\n"

cnt = codecs.escape_decode(cnt)[0].decode("utf-8")  # escape_decode返回(bytes, len(bytes))
                                                    # "仨竺汃昗宍竺訵宍泗\n藤训号\n"
```

#### 单字节编码
```python
# unicode转bytearray
multi_byte_stream = s.encode("utf-8")
# 单字节转bytearray
single_byte_stream = s.encode("latin-1")            # 字符串s里的Unicode值应都处于0~255范围内

# bytearray转unicode
unicode_str = byte_stream.decode("utf-8")           # 转化为unicode应严格符合bytearray2unicode规则
# bytearray转单字节形式
single_byte_str = byte_stream.decode("latin-1")     # 很多BBPE词表就是以该形式保存
```


#### bytearray2unicode
在字节转unicode设计中，为使互相转换无歧义，设计了以下规则：

|字节类型 |首字节前缀 |后续字节前缀 | Unicode范围  |
| --- | --- | --- |  --- |
|单字节 | `0b0xxxxxxx` | 无 |  `U+0000` ~ `U+007F` |
|双字节 | `0b110xxxxx` | `0b10xxxxxx` |  `U+0000` ~ `U+07FF` |
|三字节 | `0b1110xxxx` | `0b10xxxxxx` |  `U+0000` ~ `U+FFFF` |
|四字节 | `0b11110xxx` | `0b10xxxxxx` |  `U+10000` ~ `U+10FFFF` |

> 各字节去除前缀后的比特位 `cancat` 即为最终unicode结果

1. 前缀码系统：首字节以`0b0`开头表示该unicode是单字节字符；以`0b` + `1*n` + `0`开头表示为n字节字符
2. 后续字节标记：为避免与首字节冲突，unicode所有后续字节前两bit均以`0b10`开头
3. 严格合法性检查：decode会拒绝所有不合法的字节序列（不符合前缀规约即报错）

```python
byte_stream = bytearray()   # 存放字节码（元素值为整型）
byte_stream.decode("utf-8") # bytearray解码映射为unicode
```