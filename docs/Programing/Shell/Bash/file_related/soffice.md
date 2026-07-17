---
title: "soffice"
---

是 LibreOffice 办公套件的命令行启动器，主要用于处理文档。

### 环境部署

=== "windows"
    1. 下载[LibreOffice](https://zh-cn.libreoffice.org/download/libreoffice/)
    2. 安装
    3. 设置环境变量（soffice.exe所在的目录）

=== "linux"
    ```bash
    sudo apt-get install libreoffice-common libreoffice-writer fonts-noto fonts-noto-cjk
    ```

### `--convert-to`

`soffice --convert-to` 是 LibreOffice 提供的一个强大命令行参数，允许用户将文档批量或单独转换为其他格式。以下是对该命令的详细介绍：

基本语法为 `soffice --headless --convert-to OutputFileExtension[:OutputFilterName] [--outdir <output_dir>] source_files`

- `--headless` 无头模式，不启动任何图形界面
- `OutputFileExtension` 目标转换类型，支持pdf, docx, odt, html, txt, csv等几乎所有常见格式
- `--outdir ./` 指定输出目录，未指定时默认为当前路径
- `source_files` 待转换的文件，支持通配符，如 `*.odt`，`*.doc`

```bash
soffice --convert-to pdf moe_convolution.pptx GLM.pptx
soffice --convert-to pdf *.pptx
```

!!! info
    Windows系统下可以直接使用基于Miscrosoft的包实现转换 `from docx2pdf import convert; convert(docx_path, pdf_path)`
