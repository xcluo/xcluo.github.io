---
title: "ufw"
---

ufw（**u**ncomplicated **f**ire**w**all）是 Linux 系统上的一款简单防火墙管理工具，基于 iptables 构建，通过简化 iptables 命令，让用户可以轻松管理网络访问规则。

#### 基本语法

```bash
ufw <command>
```

Commands:

- `ufw enable` 启用防火墙
- `ufw disable` 禁用防火墙
- `ufw status` 查看防火墙状态
- `ufw status verbose` 查看详细状态（含规则编号）
- `ufw default allow` 设置默认策略为允许
- `ufw default deny` 设置默认策略为拒绝
- `ufw allow <port>[/<tcp|udp>]` 允许指定端口和协议
- `ufw deny <port>[/<tcp|udp>]` 拒绝指定端口和协议
- `ufw allow from <ip>` 允许指定 IP 地址
- `ufw deny from <ip>` 拒绝指定 IP 地址
- `ufw allow from <ip> to any port <port>` 允许特定 IP 访问特定端口
- `ufw delete <rule>` 删除指定规则
- `ufw reset` 重置所有防火墙规则
- `ufw reload` 重新加载防火墙规则

#### 应用示例

```bash
# 初始配置（按需调整）
ufw default deny                          # 默认拒绝所有入站
ufw allow 22/tcp                          # 允许 SSH
ufw allow 80/tcp                          # 允许 HTTP
ufw allow 443/tcp                         # 允许 HTTPS
ufw allow from 192.168.1.0/24 to any port 22  # 仅允许内网段 SSH

# 启用并验证
ufw enable                                # 启用防火墙（输入 y 确认）
ufw reload                                # 修改规则后重载
ufw status verbose                        # 查看完整规则列表

# 规则管理
ufw delete allow 80                       # 删除规则
ufw delete deny from 10.0.0.1             # 删除规则
ufw reset                                 # 重置所有规则
```

> 启用前务必确保 SSH（22端口）已放行，否则可能直接失去远程连接
