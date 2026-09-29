---
title: "screen"
---

screen 是 Linux/Unix 下常用的终端复用器，全称 GNU Screen。它允许你在一个终端里创建多个虚拟终端窗口，并且可以把整个会话“分离”到后台。即使 SSH 断开，里面运行的命令也会继续执行，之后可以重新连接回来。常用于远程跑长时间任务、训练模型、编译、备份等。

```bash
# 查看隔离窗口情况
screen -ls

# 进入窗口
screen -r <screen_id>
# 有则进入，无则创建窗口
screen -R <screen_id>

# 关闭/结束窗口
screen -S <screen_id> -X quit

# 创建指定名字窗口
screen -S <screen_name>
>>> <screen_id>.<screen_name>

# datached 窗口
screen -d <screen_id>
```
