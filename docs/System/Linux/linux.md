---
title: "linux"
---


### 软件相关

#### 更新源
```bash
#!/bin/bash
# change-sources.sh

echo "=== 快速更换Ubuntu软件源 ==="

# 检测当前Ubuntu版本
UBUNTU_VERSION=$(lsb_release -cs)
echo "检测到Ubuntu版本: $UBUNTU_VERSION"

# 备份源文件
sudo cp /etc/apt/sources.list /etc/apt/sources.list.backup.$(date +%Y%m%d_%H%M%S)
echo "已备份原配置"

# 选择镜像源
echo "请选择镜像源:"
echo "1) 阿里云 (mirrors.aliyun.com)"
echo "2) 清华源 (mirrors.tuna.tsinghua.edu.cn)" 
echo "3) 中科大 (mirrors.ustc.edu.cn)"
echo "4) 华为云 (repo.huaweicloud.com)"
read -p "输入选择 (1-4): " mirror_choice

case $mirror_choice in
    1) MIRROR="mirrors.aliyun.com" ;;
    2) MIRROR="mirrors.tuna.tsinghua.edu.cn" ;;
    3) MIRROR="mirrors.ustc.edu.cn" ;;
    4) MIRROR="repo.huaweicloud.com" ;;
    *) MIRROR="mirrors.aliyun.com" ;;
esac

echo "使用镜像: $MIRROR"

# 生成新的sources.list
sudo tee /etc/apt/sources.list > /dev/null << EOF
deb http://$MIRROR/ubuntu/ $UBUNTU_VERSION main restricted universe multiverse
deb http://$MIRROR/ubuntu/ $UBUNTU_VERSION-security main restricted universe multiverse
deb http://$MIRROR/ubuntu/ $UBUNTU_VERSION-updates main restricted universe multiverse
deb http://$MIRROR/ubuntu/ $UBUNTU_VERSION-proposed main restricted universe multiverse
deb http://$MIRROR/ubuntu/ $UBUNTU_VERSION-backports main restricted universe multiverse
EOF

echo "源文件更新完成!"
echo "运行 'sudo apt update' 更新软件列表"
```

#### docker

1. 更新源，编辑`/etc/apt/sources.list`

    ```bash
    deb http://ports.ubuntu.com/ubuntu-ports/ jammy main universe multiverse
    deb http://ports.ubuntu.com/ubuntu-ports/ jammy-updates main universe multiverse
    deb http://ports.ubuntu.com/ubuntu-ports/ jammy-backports main universe multiverse
    deb http://ports.ubuntu.com/ubuntu-ports/ jammy-security main universe multiverse
    ```

2. 加载更新`sudo apt update`  
3. 卸载已安装docker，`sudo apt-get remove docker docker-engine docker.io containerd runc docker-compose docker-compose-plugin`  
4. 添加docker密钥

    ```bash
    sudo apt-get update

    sudo apt-get install -y ca-certificates curl gnupg lsb-release

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    ```

5. 软件安装

    ```bash
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ```

6. 设置常态使用
    - 修改引擎源配置文件 `/etc/docker/daemon.json`  
    - 将当前用户加入docker组 `sudo usermod -aG docker $USER`
    - 重启docker服务以生效配置 `sudo systemctl restart docker`

### 远程连接

相关网络信息查看命令

```bash
ip addr     # 查看ip、子网掩码
ip route    # 查看网关
ifconfig    # 格式化查看ip、子网掩码
nmcli device show | grep DNS    # 查看DNS服务器信息
```

#### ssh远程连接

```bash
# 安装 openssh-server
sudo apt update
sudo apt install -y openssh-server

# 检查 SSH 服务状态（显示 active (running) 即为正常）
sudo systemctl status ssh

# 设置开机自启
sudo systemctl enable ssh
```


#### 远程桌面连接(暂未成功)

1. 安装xrdp(remote desktop protocol)

    ```bash
    # 更新软件源
    sudo apt update

    # 安装 xrdp 和相关依赖
    sudo apt install -y xrdp

    # 给 xrdp 权限（关键步骤）
    sudo adduser xrdp ssl-cert

    # 重启 xrdp 服务
    sudo systemctl restart xrdp
    # 查看 xrdp 服务状态
    sudo systemctl status xrdp

    # 设置开机自启
    sudo systemctl enable xrdp

    # 关闭防火墙
    sudo ufw allow 3389/tcp
    ```

=== "使用旧版Xfce桌面"

    - 切换到 Xfce 桌面
        ```bash
        sudo apt install xfce4 xfce4-goodies dbus-x11 -y

        echo "xfce4-session" > ~/.xsession
        chmod +x ~/.xsession
        ```

    - 配置xrdp使用Xfce桌面（执行内容替换）`sudo vim /etc/xrdp/startwm.sh`

        ```bash
        #!/bin/bash
        unset DBUS_SESSION_BUS_ADDRESS
        unset XDG_RUNTIME_DIR
        unset SESSION_MANAGER
        unset XAUTHORITY
        exec startxfce4
        ```  

=== "CASE: 使用ubuntu默认桌面(GNOME)"

    - 安装gnome并指定连接方式
        ```bash
        sudo apt install xrdp gnome-session gnome-shell ubuntu-desktop
        echo "gnome-session" > ~/.xsession
        ```

    - 文件配置 `sudo nano /etc/xrdp/startwm.sh`，在文件末尾 `test -x ...` 行之前添加：

        ```bash
        export GNOME_SHELL_SESSION_MODE=ubuntu
        export XDG_CURRENT_DESKTOP=ubuntu:GNOME
        export XDG_CONFIG_DIRS=/etc/xdg/xdg-ubuntu:/etc/xdg
        ```

    - 重启服务`sudo systemctl restart xrdp`

=== "CASE: 使用ubuntu默认桌面(xorgxrdp)"

    - 安装xorgxrdp
        ```bash
        sudo apt install xorgxrdp
        ```

    - `sudo vim /etc/xrdp/xrdp.ini`，确保有以下Xorg配置
        ```bash
        [Xorg]
        name=Xorg
        lib=libxup.so
        username=ask
        password=ask
        ip=127.0.0.1
        port=-1
        code=20
        ```


- `sudo vim /etc/gdm3/custom.conf` 放开`WaylandEnable=false`的注释