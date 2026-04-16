---
title: "apt"
---

| 工具 | 时代 | 特点 |
| --- | --- | --- |
| `apt-get` | 传统（1998年） | 底层、稳定、脚本友好 |
| `apt` | 现代（2014年） | 用户友好、彩色输出、进度条 |

apt（**A**dvanced **P**ackaging **T**ool），`apt-get` 与 `apt` 功能完全一致，区别在于

- 在编写 Shell 脚本或进行自动化运维时，应坚持使用 `apt-get` 系列命令，因为它们的行为更可预测，输出更稳定，不易因交互提示而中断。
- `apt` 具有更用户友好的输出格式，如提供了彩色输出、进度条、更清晰的提示信息等，更适合用户在终端进行日常交互式操作

### update

从 `/etc/apt/sources.list` 等配置文件中指定的软件源服务器，同步并下载 ==最新的软件包元数据（如软件包名称、版本号、依赖关系等）== 到本地缓存。

### install

从软件源下载并安装指定的软件包，同时会自动解析并安装该软件包所需的所有依赖项。常用语法为 `sudo apt install [OPTION] package_name[=version_number]`

OPTION

- `-y` 自动对所有提示回答“yes”
- `--only-upgrade` 仅升级已安装的软件包，如果软件未安装则不执行任何操作，避免误装。
- `--no-upgrade` 不升级已安装的软件包，如果软件已安装则不执行任何操作，避免误装。
- `--reinstall` 重新安装指定的软件包
- `-s` 在执行实际安装前，预览安装过程。
- `--no-install-recommands` 只安装核心依赖，不安装额外的推荐包。
- `--download-only` 将软件包及其所有依赖的 .deb 文件下载到 /var/cache/apt/archives/ 目录中，不进行安装

package_name格式

- `python3.10=3.10.6-1~22.04` 安装指定版本的软件
- `./example-package_1.0_amd64.deb` 安装本地软件包

    > 离线环境下不推荐使用`apt install ./package_name.deb`，因为该命令会自动联网下载相关依赖包

### upgrade

自动处理依赖关系，在某些情况下可以安装新的依赖包来完成升级。如果遇到需要删除旧包的冲突，会提示用户并等待确认，而不是直接跳过。

### remove

仅卸载软件程序本身，但会保留配置文件。常用语法为 `sudo apt remove <package_name>`

### purge

彻底卸载，不仅删除软件程序，还会删除所有相关的配置文件。`sudo apt purge <package_name>`

### clean