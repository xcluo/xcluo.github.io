---
tite: "pandoc"
---

`pandoc [OPTIONS] [FILES]`

输入输出Option

- `-f FORMAT` -r，--from，--read 指定文件输入格式
- `-t FROMAT` -w，--to，--write 指定文件输出格式
- `-o FILE` --output 指定输出文件的路径和名称

文档结构和元素据Option

- `--toc[=true|false]` --table-of-contents，在文档中生成目录。
- `--toc-depth=NUMBER` 设置目录深度（只显式到NUMBER级目录）
- `-M KEY:[VALUE]` --metadata 设置文档元数据，如标题
- `--metadata-file=FILE` 一个 YAML 或 JSON 文件中读取元数据

模板与样式Opion

- `--reference-doc=FILE` 指定内容样式和页面设置参考文件，只用于docx、pptx以及odt等二进制办公文档。通过“偷取”参考文档中的样式定义来格式化输出（不参考内容，只复制内容样式和页面设置）
- `--template=FILE` 用于 HTML、LaTeX、EPUB、文本等基于标记语言的格式（不能用于docx文件），通过替换模板文件中的占位符来生成整个文档结构。
