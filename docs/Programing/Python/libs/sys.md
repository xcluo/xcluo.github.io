```python
import sys
```
#### 导入本地库

```python
sys.path.append(absolute/relative_project_path)  # project路径

# 1. 不同项目时推荐使用绝对路径
# 2. 不同项目时最好别涉及父路径(可能重名)，一步到位.py文件目录
import .<python_file_name>
# or
from .<python_file_name> import <class_name or *>
```
> `...`表示回退三层