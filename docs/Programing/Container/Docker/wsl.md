---
title: WSL
---
WSL (Windows Subsystem for Linux)

#### wsl基础命令

- powershell命令查看所有分发版存储路径
    ```bash
    Get-ChildItem HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss\ | ForEach-Object {
        $props = Get-ItemProperty $_.PSPATH
        [PSCustomObject]@{
            名称 = $props.DistributionName
            路径 = $props.BasePath
            版本 = $props.Version
        }
    }
    ```
    > 分发版文件默认存放在`C:\Users\LXC\AppData\Local\wsl\{...}/ext4.vhdx` 中

- `wsl --list --verbose, wsl -l -v`，显示所有分发版信息
- `wsl --shutdown`，关闭wsl服务
- `wsl --set-default <发行版名称>`，设置默认wsl虚拟机
- `wsl -d <发行版名称>`，切换到指定wsl虚拟机

#### 修改wsl分发版位置

1. 备份分发版数据文件 `wsl --export <发行版名称> D:\ubuntu.tar` 
2. 删除wsl分发版 `wsl --unregister <发行版名称>` 
3. 从备份恢复分发版（版本2）并指定位置 `wsl --import <发行版名称> D:\wsl\ D:\ubuntu.tar --version 2` 

#### 更新wsl 账户密码
1. 管理员权限进入wsl
2. `wsl -u root`
3. `passwd <user_name>`     # 修改指定用户密码
4. 重复输入两次密码进行更改