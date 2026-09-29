---
title: "vim"
---

**vim操作**

```
gg                      # 跳转到首行
G                       # 跳转到末行
ESC + / + <content>     # 内容查找，使用n切换下一个，N切换至上一个
                        # content中含关键字/时需要使用转义字符\/表示
Ngg
NG
:N # 跳转到指定行

# 后N行
Nj
:+N

# 前N行
Nk
:-N
```

#### vim配置

于`~/.vimrc`文件编辑，无文件新建即可

```bahsh
set ts=4            # tab=4个空格, 缺省为8个空格
set tabstop=4       # tab=4个空格宽度，缺省为8个空格宽度
set nu              # 设置vim显示行号，缺省不显示
set nonu            # 等价于set nu!   即设置vim不显示行号
set mouse=r         # 可以直接将外部内容复制入vim编辑器窗口, 不行的话设置为set mouse=v
set mouse=a         # 缺省状态，即不能直接将外部内容复制入vim编辑器窗口
```
