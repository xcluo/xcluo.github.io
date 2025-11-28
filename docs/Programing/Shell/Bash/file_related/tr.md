---
title: "tr"
---

### 基本语法
```bash
tr [options] SET1 [SET2]
```
tr (transliterate转写) 从标准输入读取数据，将属于字符集 SET1 的字符替换为字符集 SET2 中对应位置的字符，然后输出到标准输出。

#### options
- `-d` 删除字符
- `-c` 对给定字符集执行补集操作
- `-s` 压缩字符集中各连续重复字符数量至1
- `-t` 当`#!python len(SET1) > len(SET2)`时，执行截断操作`#!python SET1 = SET2[:len(SET2)]`
    
    !!! info ""
        当未使用`-t`选项，且`#!python len(SET1) != len(SET2)`时  

        - `#!python len(SET1) > len(SET2)` → 执行补齐操作`#!python SET2 = SET2 + SET2[-1]*(len(SET1) - len(SET2))`  
        - `#!python len(SET1) < len(SET2)` → 执行补齐操作`#!python SET2 = SET2[:len(SET1)]`

#### SET
集合类似于正则表达式，只是取消了左右两边的方括号

- `[:alum:]` 字母和数字，等价于`a-zA-Z0-9`
- `[:alpha:]` 字母，等价于`a-zA-Z`
- `[:digit:]` 数字，等价于`0-9`
- `[:lower:]` 小写字母，等价于`a-z`
- `[:upper:]` 大写字母，等价于`A-Z`
- `[:space:]` 空白字符
- `[:punct:]` 标点符号
#### 常用方法
```bash
# 替换 + 压缩，将A替换为B后再对结果进行压缩
tr -s "A" "B"

# 大写转小写 | 将空格换成换行 + 压缩 | sort + 统计各单词数量
echo "Hello hello \nWORLD world" | tr '[:upper:]' '[:lower:]' | tr -s ' ' '\n' | sort | uniq -c
```