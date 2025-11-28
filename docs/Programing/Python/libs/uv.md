---
title: "uv"
---

`pip install uv`

### 创建虚拟环境
1. `uv env` 创建.env虚拟环境
2. `uv venv --python 3.11` 创建虚拟环境，且指定python版本
3. `uv venv --python /path/to/python3.11` 创建虚拟环境光，并指定路径中的python解释器

### 使用、推出虚拟环境
1. `source .venv/bin/activate` 进入虚拟环境
    - `which python` 查看当前python解释器
    - 无需激活使用虚拟环境`uv run python main.py`
2. `deactivate` 退出虚拟环境