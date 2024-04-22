`Aho–Corasick`算法是由Alfred V. **Aho**和Margaret J.**Corasick** 发明的字符串搜索算法，用于在输入的一串字符串中匹配有限组字典树中的子串。
#### 使用方法
```python
from ahocorasick import Trie
trie = Trie(
    allow_overlaps=True,            # False：只返回匹配字典树最长的关键字
                                    # True：允许返回所有匹配的关键字
    case_sensitive=False            # 是否大小写敏感
)
trie.build_trie(keyword_list)       # 向字典树中添加关键字词典
emits = trie.parse_text(text)       # 基于既有关键字词典和ahocorasick算法实现关键字匹配
                                    # emit.start, emit.end, emit.keyword

def replace_span(text):
    intervals = trie.parse_text(text)
    text_new = []
    idx = 0
    
    for interval in intervals:
        text_new += text[idx: interval.start]
        span = text[interval.start: interval.end + 1]
        replace_span = replace_map[span]
        text_new.append(replace_span)
        idx = interval.end + 1

    text_new.append(text[idx:])
    return "".join(text_new)
```
