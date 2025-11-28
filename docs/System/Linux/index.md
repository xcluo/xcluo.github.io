1. 更新源
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

- 修改firefox默认分辨率
    1. 网址栏输入 `about:config`，确定风险
    2. 搜索 `layout.css.devPixelsPerPx`，1.2即为120%
