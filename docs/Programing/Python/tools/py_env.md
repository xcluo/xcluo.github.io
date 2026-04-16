

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

## uv

- 下载程序 [uv release](https://github.com/astral-sh/uv/releases)
- 解压，将uv文件夹添加到环境变量中

### 创建虚拟环境

1. `uv venv PATH=.venv` 在指定路径创建虚拟环境
2. `uv venv --python 3.11` 创建虚拟环境，且指定python版本
3. `uv venv --python /path/to/python3.11` 创建虚拟环境光，并指定路径中的python解释器
4. `uv run .` 自动根据文件pyproject.toml处理虚拟环境和依赖安装，然后尝试启动项目。

### 使用、退出虚拟环境

1. `source .venv/Scripts/activate` 激活虚拟环境（类似于conda activate）
    - `which python` 查看当前python解释器
    - 无需激活使用虚拟环境`uv run python main.py`
2. `deactivate` 退出虚拟环境

#### 搭建

```bash
# （若无）创建pyproject.toml用于管理依赖
uv init
```

```bash
uv add "requests>=2.28.0"
uv add "requests==2.31.0"
uv add "requests~=2.28.0"   # 兼容版本
uv pip install -e .         # 使用uv pip兼容 Poetry 格式安装
# 指定 uv pip install -i 指定源
-i https://pypi.mirrors.ustc.edu.cn/simple/
-i https://pypi.tuna.tsinghua.edu.cn/simple

uv sync                     # 根据 pyproject.toml 或 requirements.txt 同步虚拟环境，安装/卸载所有依赖
uv remove pip               # 卸载依赖包

uv export                   # 将 pyproject.toml 中的依赖导出为其他格式（如 requirements.txt）
uv pip list                 # 列出 uv 已下载/管理的所有 python 包
uv python list              # 列出 uv 已下载/管理的所有 Python 版本
uv python install <version> # 安装指定 Python 版本
``` 

## poetry
