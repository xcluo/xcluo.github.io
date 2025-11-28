---
title: "seq"
---

### seq

#### 基本语法
```bash
seq [option]... last
seq [option]... first last
seq [option]... first step last
```

- `-f` --format=FORMAT，指定浮点型数据输出格式
- `-s str` --separator=STRING，指定序列分隔字符，默认为`\n`
- `-w` --equal-width，指定等宽输出（不足使用前导0填充），默认为false

#### 常用方法
```bash
seq -s ", " 1 5                 # 1, 2, 3, 4, 5
for i in $(seq -w 1  2 10); do  # 01 03 05 07 09
```

### {start..end(..step)}
`step`表示前后两个元素的步长，默认为1，当`start > end`时，表示递减步长。

#### 常用方法
```bash
echo {10..1..2}         # 10 8 6 4 2
echo {01..10..2}        # 01 03 05 07 09
echo {a..z..5}
```

### for循环


```bash
for (( i=1; i<=5; i++ )); do

# 复杂表达式
for (( i=0, j=10; i<j; i++, j-- )); do

# 使用变量
start=5
end=15
step=3
for (( i=start; i<=end; i+=step )); do

# 预定义数组，无分隔符
fruits=("apple" "banana" "orange" "grape")
for fruit in "${fruits[@]}"; do
```