grep (**G**lobal search **R**egular **E**xpression and **P**rint out the line)，利用正则表达式全局搜索并打印目标行

### `grep`

#### options
`--color` 彩色显示   
`-i` 忽略大小写    
`-v` 反向搜索,不打印匹配的行  



#### 多条件
1. or
```bash
# 搜索包含关键字 【'apple'】 或 【以关键字'banana'结尾】 的行
grep 'apple\|banana$' file.txt
```
1. and
```bash
# 搜索包含关键字 【'apple'】 且 【以关键字'banana'结尾】 的行
grep -e 'apple' -e 'banana$' file.txt
```
1. not
```bash
# 搜索不包含关键字 【'apple'】 的行
grep -v 'apple' file.txt
```

### `egrep`

#### `fgrep`