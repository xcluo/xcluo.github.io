---
title: "条件语句"
---
### 条件表达式

#### 文件相关

基本语法 `#!bash [ Option $var ]`

Option：

- `-e` 是否存在文件或路径
- `-d` 是否为目录
- `-f` 是否为普通文件（非目录、非设备文件）
- `-x` 是否有可执行权限
- `-r` 是否有读取权限
- `-w` 是否有写入权限
- `-s` 是否为空文件

#### 字符串相关

1. 单目运算符基本语法：`#!bash [ Option $var ]`

    Option：

    - `-z` 是否为空字符串，即长度是否为zero
    - `-n` 是否非空字符串，即长度是否为non-zero

2. 双目运算符基本语法：`#!bash [ $var1 Option $var2 ]`

    Option：

    - `==` 左字符串是否匹配右字符串
    - `!=` 左字符串是否不匹配右字符串
    - `=~` 右字符串是否为左字符串字串

    > 字符串运算基于模式匹配机制，因此支持通配符

#### 数值相关

1. 单中括号基本语法：`#!bash [ $var1 Option $var2 ]`

    Option：

    - `-eq` 数值相等
    - `-ne` 数值不等
    - `-lt` 左值小于右值
    - `-le` 左值小于等于右值
    - `-gt` 左值大于右值
    - `-ge` 左值大于等于右值

2. 双圆括号基本语法：`(( var1 Ops var2))`

    - 此时无需对变量进行 `$var` 转义
    - 常用的运算符有 `==, !=, <, <=, >, >=`
    - 支持中间变量操作 `(( var1 + 10 < var2 ))`

#### 组合逻辑表达式

1. 单中括号逻辑运算符基本语法：`[ Option expr ]`
    - `-a` 逻辑与
    - `-o` 逻辑或
    - `!` 逻辑非

2. 双中/圆括号括号逻辑运算符基本语法：`(( Ops expr ))`、`[[ Ops expr ]]`
    - `&&` 逻辑与
    - `||` 逻辑或
    - `!` 逻辑非

### 分支语句

#### if-elif-else-fi

```bash
if cond1; then
    ...
elif cond2; then
    ...
else
    ...
fi
```

#### case-esac

```bash
case $var in 
    pattern_1)              # $var 匹配上 pattern_1
        ...
    ;;
    
    pattern_2|pattern_3)    # $var 匹配上 pattern_2 或 pattern_3
        ...
    ;;

    pattern*)               # $var 以pattern开头
        ...
    ;;
    *)                      # other
        ...
    ;;
esac
```

常用通配符如下：

- `*` 匹配 0 个或多个任意字符
- `?` 匹配单个任意字符
- `[]` 匹配给定候选集中任意单个字符，如`[0-9], [a-z], [abc]`

### 循环语句

#### for

```bash
for var in sequence
do
    ...
done
```

#### while

```bash
while cond
do
    ...
done
```

#### until

执行循环体直到cond条件成立才退出循环

```bash
until cond
do
    ...
done
```

### 循环控制语句

#### break

#### continue
