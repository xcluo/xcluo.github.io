---
title: "rpm"
---

## rpm

rpm（**R**ed Hat **P**ackage **M**anager）是 Red Hat 系 Linux 发行版（如 RHEL、CentOS、Fedora）中最基础的底层包管理工具。基本语法为 `rpm [OPTIONS]`

Option

- `-i pkg.rpm` --install 安装
- `-U pkg.rpm` --upgrade 升级安装
- `-e pkg.rpm` --erase 卸载
- `-q` 查询（query）
- `-V` --verify 校验
- `-v` --verbose 显示详细过程
- `-h` --hash 显示安装进度条
- `--nodeps`  不检查依赖关系
- `--force`  强制安装/升级
- `--test`  测试安装，不实际执行

### Option

=== "安装-i"

    ```bash
    rpm -i package.rpm
    rpm -ivh package.rpm            # 显示详细信息
    rpm -ivh --nodeps package.rpm   # 不检查依赖直接安装（慎用）
    rpm -ivh --force package.rpm    # 强制安装（可覆盖已安装的包）
    ```

=== "查询-q"

    ```bash
    rpm -q package_name         # 查询包是否已安装
    rpm -qa                     # 查询所有已安装的包
    rpm -qi package_name        # 查询已安装包的信息
    rpm -ql package_name        # 查询包的文件列表
    rpm -qf /path/to/file       # 查询文件属于哪个包
    rpm -qc package_name        # 查询包的配置文件
    rpm -qR package_name        # 查询包的所有信息（包含依赖关系）
    ```

=== "卸载-e"

    ```bash
    rpm -e package_name             # 卸载包
    rpm -e --nodeps package_name    # 不检查依赖直接卸载
    ```

=== "升级-U"

    ```bash
    rpm -Uvh package.rpm                # 升级包（自动卸载旧版本）
    rpm -Uvh --oldpackage package.rpm   # 升级但不改变配置文件
    ```

=== "校验-V"

    ```bash
    rpm -V package_name                 # 校验已安装包的文件完整性
    rpm -Va                             # 校验所有包
    rpm --import /path/to/RPM-GPG-KEY   # 导入公钥（用户对安装包验证签名）
    ```

!!! info
    1. **依赖问题**：rpm 本身不会自动下载依赖，如果遇到依赖缺失，需要手动处理或者使用yum 或 dnf 来自动处理依赖
    2. **版本冲突**：安装前确认是否已有旧版本，若存在可能需要先卸载再安装

## yum

yum（**Y**ellowdog **U**pdater, **M**odified）是 Red Hat 系 Linux 发行版（如 RHEL、CentOS、Fedora）中非常经典的上层包管理工具，能够自动处理依赖关系。基本语法为 `yum COMMAND [OPTIONS] [packages]`

Command

- `install` 安装包
- `remove` 卸载包
- `update` 升级包
- `list` 列出包
- `search` 搜索包
- `info` 查看包信息
- `clean` 清理缓存
- `repolist` 列出仓库
- `groupinstall` 安装包组
- `history` 查看历史

### 基础操作

=== "安装"

    ```bash
    yum install package_name          # 安装单个包（自动处理依赖）
    yum install -y package_name       # 非交互模式，自动确认
    yum localinstall package.rpm      # 安装本地 rpm 包并自动处理依赖
    yum reinstall package_name        # 重新安装
    ```

=== "卸载"

    ```bash
    yum remove package_name           # 卸载包（同时移除依赖）
    yum erase package_name            # 同 remove
    ```

=== "升级"

    ```bash
    yum update                # 升级所有可更新的包
    yum update package_name   # 升级指定包
    yum check-update          # 检查可更新的包（不执行更新）
    ```

### 查询搜索

```bash
yum list                       # 列出所有可用的包
yum list installed             # 列出所有已安装的包
yum list available             # 列出所有可安装的包
yum list updates               # 列出所有可更新的包
yum search keyword             # 搜索包（按名称和描述）
yum search all keyword         # 搜索所有匹配项
yum info package_name          # 查看包详细信息
yum provides /path/to/file     # 查询文件属于哪个包
yum whatprovides filename      # 同 provides
yum deplist package_name       # 列出包的所有依赖
```

### 仓库管理

```bash
yum repolist                    # 列出所有已启用的仓库
yum repolist all                # 列出所有仓库（包括禁用的）
yum-config-manager --add-repo http://example.com.repo   # 添加仓库
yum-config-manager --enable repo_name    # 启用指定仓库
yum-config-manager --disable repo_name   # 禁用指定仓库
yum makecache         # 重建 yum 缓存
```

### 组管理

```bash
yum groupinstall "Group Name"     # 安装软件包组
yum groupremove "Group Name"      # 卸载软件包组
yum groupupdate "Group Name"      # 升级软件包组
yum groups list                   # 列出所有软件包组
yum groups info "Group Name"      # 查看软件包组详情
```

### 清理缓存

```bash
yum clean all           # 清理所有缓存（headers、packages、metadata 等）
yum clean headers       # 只清理 headers 缓存
yum clean packages      # 只清理 packages 缓存
yum clean metadata      # 只清理 metadata 缓存
yum makecache           # 重建缓存
```

### 仅下载

yumdownloader 是 yum-utils 包提供的工具，用于仅下载 rpm 包而不安装。

```bash
yumdownloader package_name              # 下载单个包到当前目录
yumdownloader --destdir /path/to/dir package_name   # 下载到指定目录
yumdownloader --resolve package_name    # 同时下载依赖包
yumdownloader --archlist x86_64 package_name        # 下载指定架构的包
yumdownloader package1 package2         # 下载多个包
```

| 选项 | 说明 |
|------|------|
| `--destdir <dir>` | 指定下载目录 |
| `--resolve` | 同时下载所需依赖 |
| `--archlist <arch>` | 指定目标架构（见下表） |
| `--urls` | 仅显示下载 URL，不实际下载，但需指定--destdir |

**常见架构取值**：

| 架构 | 说明 |
|------|------|
| `x86_64` | 64 位 x86（Intel/AMD） |
| `i686` / `i586` / `i386` | 32 位 x86 |
| `aarch64` | 64 位 ARM（ARMv8） |
| `armhfp` | 32 位 ARM（ARMv7 硬浮点） |
| `ppc64le` | 64 位 PowerPC 小端 |
| `ppc64` | 64 位 PowerPC 大端 |
| `s390x` | IBM Z 系统 |
| `noarch` | 架构无关包（Python、Perl 等） |
| `src` | 源码包（SRPM） |

多个架构可逗号分隔，例如：`--archlist x86_64,i686`

!!! note
    - 需先安装 `yum-utils` 包：`yum install yum-utils`
    - 下载的包会保存在当前目录或指定目录，可用于离线安装

!!! tip
    - 使用 `yum history` 可以查看安装历史，并可通过 `yum history undo <id>` 回滚操作

## dnf

dnf（Dandified YUM）是 YUM 的下一代替代品。从 RHEL 8 和 Fedora 22 开始，DNF 正式取代 YUM 成为默认的包管理工具。它提供了更快的性能、更高的内存效率和更强大的依赖关系解析能力。基本语法为 `dnf <command> [options] [packages]`

Command

- `install` 安装包
- `remove` 卸载包
- `upgrade` 升级包
- `list` 列出包
- `search` 搜索包
- `info` 查看包信息
- `clean` 清理缓存
- `repolist` 列出仓库
- `groupinstall` 安装包组
- `history` 查看历史
- `module` 模块流管理
- `download` 仅下载包
- `updateinfo` 更新信息

dnf 的命令与 yum 几乎兼容，以下是主要命令对照：

### 基础操作

=== "安装"

    ```bash
    dnf install package_name          # 安装单个包
    dnf install -y package_name       # 非交互安装
    dnf localinstall package.rpm      # 安装本地包
    dnf reinstall package_name        # 重新安装
    dnf upgrade package_name          # 升级指定包
    ```

=== "卸载"

    ```bash
    dnf remove package_name           # 卸载包
    dnf erase package_name            # 同 remove
    ```

=== "升级"

    ```bash
    dnf upgrade                # 升级所有包
    dnf upgrade --security     # 仅安装安全更新
    dnf check-update           # 检查可更新的包
    ```

### 查询搜索

```bash
dnf list                       # 列出所有可用包
dnf list installed             # 列出已安装的包
dnf list available             # 列出可安装的包
dnf list extras                # 列出已安装但不在仓库中的包
dnf list recently              # 列出最近新增的包
dnf search keyword             # 搜索包
dnf info package_name          # 查看包详情
dnf provides /path/to/file     # 查询文件属于哪个包
dnf repoquery --requires package_name   # 查询包的依赖
dnf repoquery --alldeps package_name    # 查询包的所有依赖（包括传递依赖）
```

### 仓库管理

```bash
dnf repolist                    # 列出已启用的仓库
dnf repolist --all              # 列出所有仓库
dnf module reset module_name    # 重置模块流
dnf module enable module_name   # 启用模块流
dnf module disable module_name  # 禁用模块流
dnf module list                 # 列出所有模块流
dnf makecache                   # 重建缓存
```

### 组管理

```bash
dnf groupinstall "Group Name"     # 安装软件包组
dnf groupremove "Group Name"      # 卸载软件包组
dnf groupupdate "Group Name"      # 升级软件包组
dnf groups list                   # 列出软件包组
dnf groups info "Group Name"      # 查看软件包组详情
dnf groups install "Group Name"   # 安装软件包组（新版命令）
```

### 清理缓存

```bash
dnf clean all           # 清理所有缓存
dnf clean metadata      # 清理 metadata 缓存
dnf clean packages      # 清理 packages 缓存
dnf makecache           # 重建缓存
```

### 特有功能

```bash
dnf history                     # 查看操作历史
dnf history undo <id>           # 回滚指定操作
dnf history info <id>           # 查看历史详情
dnf download package_name       # 仅下载包（不安装）
dnf updateinfo                  # 检查更新信息
dnf updateinfo list             # 列出所有更新公告
dnf module info module_name     # 查看模块信息
```

!!! tip
    - dnf 默认使用 `/etc/yum.repos.d/` 目录下的 repo 文件，兼容 yum 配置
    - `dnf swap old_package new_package` 可在不卸载旧包的情况下切换到新包
    - dnf 的 `dnf-2` 底层工具提供更细粒度的控制能力

!!! info "yum 与 dnf 的主要区别"

    | 特性 | yum | dnf |
    |------|-----|-----|
    | 默认版本 | RHEL/CentOS 7 及以下 | RHEL 8+, Fedora 22+ |
    | 依赖解析 | 较慢 | 更快、更智能 |
    | 内存占用 | 较高 | 更低 |
    | 葵花锁（Lock） | 不支持 | 支持 `--lock` 选项 |
    | 模块流 | 不支持 | 支持 |
