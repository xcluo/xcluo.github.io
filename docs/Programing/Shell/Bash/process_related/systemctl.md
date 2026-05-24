---
title: "systemctl"
---

systemctl（**system**d **c**on**t**ro**l**‌） 是 systemd 初始化系统的主要管理工具，用于控制系统和服务管理器。它取代了传统的 SysV init 脚本管理方式（service、chkconfig 等），几乎在所有主流 Linux 发行版（如 CentOS/RHEL 7+、Fedora、Ubuntu 16.04+、Debian 8+、Arch Linux）中通用。

#### 基本语法

```bash
systemctl [OPTIONS...] COMMAND [UNIT...]
```

Options:

Command

- `start` 启动服务
- `stop` 终止服务
- `restart` 重启服务
- `enbale` 设置开机自启
- `disable` 取消开机自启
- `is-enabled` 检查是否开机自启
- `mask` 屏蔽服务（禁止任何方式启动，包括手动）
- `unmask` 取消屏蔽

Unit：systemd 中管理的对象名称，常见的有

- `nginx`
- `docker`
- `mysql`