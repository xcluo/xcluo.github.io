grep (**g**lobal search **r**egular **e**xpression and **p**rint out the line)，利用正则表达式全局搜索并打印目标行

### `grep`

#### options
- `--color` 彩色着重显示匹配部分
- `-i` 正则匹配忽略大小写
- `-v` 条件取反


#### 多条件
1. or
```bash
# 搜索包含关键字 【'apple'】 或 【以关键字'banana'结尾】 的行
grep 'apple\|banana$'
grep -e 'apple' -e 'banana$'
```
1. and
```bash
# 搜索包含关键字 【'apple'】 且 【以关键字'banana'结尾】 的行
grep 'apple' | grep 'banana$'
```
1. not
```bash
# 搜索不包含关键字 【'apple'】 的行
grep -v 'apple'
```

### `egrep`

#### `fgrep`