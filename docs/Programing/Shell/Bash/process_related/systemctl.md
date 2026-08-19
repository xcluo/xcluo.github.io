---
title: "systemctl"
---

## 基本语法

systemctl（**system**d **c**on**t**ro**l**‌） 是 systemd 初始化系统的主要管理工具，用于控制系统和服务管理器。基本语法为 `systemctl [OPTIONS ...] COMMAND ...`

Options:

Command

- `start` 启动服务
- `stop` 终止服务
- `reload` 重新加载服务
- `daemon-reload` 重新加载 systemd 管理器自身的配置，/etc/systemd/system/
- `restart` 重启服务
- `enbale` 设置开机自启
- `disable` 取消开机自启
- `is-enabled` 检查是否开机自启
- `mask` 屏蔽服务（禁止任何方式启动，包括手动）
- `unmask` 取消屏蔽
- `list-units` 列出当前系统上所有已加载（loaded）的 systemd 单元（units）的状态信息。`systemctl list-units --type=service --state=running`

```bash
.service
.target
.socket
```
