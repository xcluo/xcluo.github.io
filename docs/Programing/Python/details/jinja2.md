Jinja2 是 Python 中最流行的模板引擎之一，广泛用于 Web 开发（如 Flask、Django）、文档生成、配置模板等场景。

```python
from jinja2 import Environment, FileSystemLoader, Template
```

### 设计模板
### 生成模板
```python
# 加载模板文件
env = Environment(
    loader=FileSystemLoader(
        searchpath,             # Union[str, list[str]]，模板文件所在目录
        )
)
template = env.get_template(
    name,                       # 模板文件名
)

# 传入参数渲染prompt，返回str
# 若需要传入json形式，可搭配json.dumps一起使用
prompt = template.render(*args, **kwargs)
```