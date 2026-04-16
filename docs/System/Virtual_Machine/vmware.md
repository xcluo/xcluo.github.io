1. 网络模式
    - 桥接模式: 虚拟机直接使用宿主机物理网络.
    - NAT模式: 虚拟机通过宿主机共享上网. 默认推荐，上网方便
    - 仅主机模式: 虚拟机与宿主机私有网络.

### NAT模式有网


1. `编辑 → 虚拟网络编辑器 → (管理员模式)更改设置`
2. `VMnet8 → 子网IP(要求和宿主机处于不同网段) → DHCP设置(尾数不为.1, 如设置为.2)`
3. 将网络连接中VMnet8的IP设置为`子网IP` +`.1`
4. 进入linux系统，手动设置有线连接IP地址（应进入虚拟网络编辑器中的DHCP设置查看分配的IP网段手动设置IP），子网掩码为24，网关为`.2`的IP地址

- [虚拟机共享文件](https://zhuanlan.zhihu.com/p/650638983)
- linux `sudo vim /etc/resolv.conf`，增加`nameserver 网关ip` 实现上网功能


#### 要启动VMware HDCP Service & VMware NAT Service

==没网的时候首先确认VMware相关网络服务是否开通==


- 需重启NAT服务（取消勾选，再勾选）
- 关闭虚拟机


#### 共享剪切板

```bash
# 更新源
sudo apt update
# 安装核心+桌面增强（必须装desktop）
sudo apt install open-vm-tools open-vm-tools-desktop -y
# 重启生效
sudo reboot
```

#### 共享文件
1. 宿主机：虚拟机 → 设置 → 选项 → 共享文件夹
2. 虚拟机：
    - `vmware-hgfsclient` → 显示共享文件夹信息  
    - `sudo mkdir /mnt/hgfs` 创建挂件文件夹  
    - `sudo /usr/bin/vmhgfs-fuse .host:/ /mnt/hgfs -o allow_other -o uid=1000 -o umask=022` 设置所有用户可访问  
    > 可直接在`/etc/fstab`中配置 `.host:/ /mnt/hgfs fuse.vmhgfs-fuse allow_other,uid=1000,gid=1000,umask=022 0 0` 以实现挂载自启动


#### 扩容
```bash
sudo apt install gparted
sudo gparted
resize → 修改new size至最新大小
```