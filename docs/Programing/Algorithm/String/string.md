#### KMP
D.E.**K**nuth，J.H.**M**orris和V.R.**P**ratt提出的高效字符串匹配算法，该算法的核心是利用匹配失败后的信息，尽量减少模式串（子串）与主串的匹配次数

- 为模式串构建`#!py next`数组，`#!py next[i]`表示当模式串的第i个元素与目标串匹配失败时，快速使用模式串的第`next[i]`个元素与目标串元素进行匹配
- 匹配失败时，**模式串右移**距离=失配字符所在位置-失配字符对应的next值
> 复杂度为$O(m+n)$

```python
def get_next(pattern_text):
    n = len(pattern_text)
    k, j = -1, 0                # k: next数组值
                                # j: pattern_text index
    next = [-1] * n
    while j < n-1:
        # 起始时刻或下一元素匹配成功
        if k == -1 or pattern_text[j] == pattern_text[k]:
            k += 1
            j += 1
            next[j] = k
        else:
            k = next[k]
    return next

def kmp_str(text, pattern_text):
    i, j = 0, 0
    next = get_next(pattern_text)
    max_common_len = 0
    match_index = -1
    while i < len(text) and j < len(pattern_text):
        if j == -1 or text[i] == pattern_text[j]:
            i += 1
            j += 1
        else:
            j = next[j]
        max_common_len = max(max_common_len, j)
        if match_index == -1 and j == len(pattern_text):
            match_index = i - j
    return match_index, max_common_len
```