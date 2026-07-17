---
title: pdf2image
---

pdf2image 是一个专注于将 PDF 文档页面转换为图像的专业 Python 库。它简单易用，核心功能就是调用系统底层的 Poppler 工具，为你返回方便处理的 PIL 图像列表，相关依赖如下

```bash
pip install pdf2image
# 安装 Poppler 工具
sudo apt-get install poppler-utils
```

### 核心函数

#### convert_from_path

```python
def convert_from_path(
    pdf_path: Union[str, PurePath],
    dpi: int = 200,                     # 输出图像质量
    output_folder: Union[str, PurePath] = None,
                                        # 直接将图片存入磁盘，防止内存爆炸
    first_page: int = None,             # [从第几页开始转化，从1开始
    last_page: int = None,              # ]到第几页结束转化
    fmt: str = "ppm",                   # 输出图像格式，{ppm, }
    thread_count: int = 1,              # 转化并行线程数
    grayscale: bool = False,            # 灰度输出
    size: Union[Tuple, int] = None,     # 调整图片尺寸
) -> List[Image.Image]:
```
