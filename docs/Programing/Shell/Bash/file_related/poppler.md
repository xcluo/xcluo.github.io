---
title: "Poppler"
---

专注于静态文档。它的核心是解析“页面布局与矢量图形”（如文字、线条、字体嵌入），并将 PDF 内部的复杂指令“画”成一张图片或提取出其中的文本。

### 环境部署

=== "windows"
    1. 下载[poppler-windows release.zip](https://github.com/oschwartz10612/poppler-windows/releases)
    2. 设置环境变量（Library/bin目录）
=== "linux"
    `sudo apt install -y poppler-utils`

### PDF转图片

pdftoppm
pdftocairo

### PDF提取图片

pdfimages
