---
title: "py_env"
---

```bash
pip cache info      # 查看缓存信息（一般存储在c盘）
pip cache list      # 查看具体缓存情况
pip cache purge     # 清除缓存
# 自定义pip缓存目录
export PIP_CACHE_DIR=/path/to/custom/cache


uv cache size       # 查看缓存大小，单位B
uv cache clean      # 清除所有缓存
uv cache prune      # 清除过期缓存
# 自定义uv缓存目录
export UV_CACHE_DIR=/path/to/custom/cache
```

## conda

- 安装后cmd中运行 `conda init` 以永久化支持conda
- 安装后git中运行 `conda init bash` 以永久化支持conda


常用安装命令

```bash
# 完全绕过 pip 的缓存机制，强制重新从网络下载
pip install --no-cache-dir package[==version]
```

## uv

- 下载程序 [uv release](https://github.com/astral-sh/uv/releases)
- 解压，将uv文件夹添加到环境变量中

### 环境操作

#### 创建虚拟环境

创建Python虚拟环境，基本语法为 `uv venv [OPTION] [PATH=.venv]`

Option

- `-p <PYTHON>` --python 指定python版本或python解释器路径

```bash
uv venv my_venv                         # 创建my_venv虚拟环境
uv venv -p 3.10                         # 指定虚拟环境的python版本为3.10.x
uv venv -p local/path/to/python3.10     # 指定本地python解释器为虚拟环境的python版本
```

#### 激活/退出

1. `source .venv/Scripts/activate` 激活虚拟环境（类似于conda activate）
    - `which python` 查看当前python解释器
2. `deactivate` 退出activate项以退出虚拟环境

### 搭建虚拟环境

#### `uv init`

项目初始化，（若无则）自动生成pyproject.toml文件，用于管理依赖

#### `uv sync`

按照灵活地根据已有的pyproject.toml或requirements.txt文件内容，自动安装依赖包
> 基于requirements.txt安装时，会自动删除不在该文件中存在的包，因此推荐使用`uv pip install -r requirements.txt`安装

#### `uv add`

`uv add [OPTIONS] <PACKAGES|-r <REQUIREMENTS>>`

Option

- `-r <REQUIREMENTS>` --requirements，指定批量依赖文件
- `-i <INDEX_URL>` --index-url，指定一个临时的包索引镜像源
- `-U` --ungrade，允许升级已存在但被锁定的包，确保获取其最新版本

Package

- 最新兼容版本：`uv add requests`
- 指定版本：`uv add requests==2.31.0`
- 指定版本范围
    - `uv add "requests>=2.31.0,<2.33.0"` 指定上下边界（可只指定单边）
    - `uv add "requests^2.31.0"` 指定大版本号，即 `≥2.31.0, <3.0.0`
    - `uv add "requests~=2.31.0"` 指定大版本号和小版本号，即 `≥2.31.0, <2.32.0`

#### `uv remove/uv pip uninstall`

- [ ] uv add, uv pip install；uv remove uv pip uninstall；pip 是临时工具，不对包管理文件做修改

```bash
uv add "requests>=2.28.0"
uv add "requests==2.31.0"
uv add "requests~=2.28.0"   # 兼容版本
uv pip install -e .         # 使用uv pip兼容 Poetry 格式安装
# 指定 uv pip install -i 指定源
-i https://pypi.mirrors.ustc.edu.cn/simple/
-i https://pypi.tuna.tsinghua.edu.cn/simple

uv remove pip               # 卸载依赖包

uv export                   # 将 pyproject.toml 中的依赖导出为其他格式（如 requirements.txt）
uv python list              # 列出 uv 已下载/管理的所有 Python 版本
uv python install <version> # 安装指定 Python 版本
``` 

1. `uv run .` 自动根据文件pyproject.toml处理虚拟环境和依赖安装，然后尝试启动项目。
- 无需激活使用虚拟环境`uv run python main.py`


## poetry
