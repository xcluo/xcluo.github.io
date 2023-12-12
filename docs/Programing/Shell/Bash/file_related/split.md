

#### 语法格式
`split [option] ... file_name [new_file_prefix]`

参数option

- `-b storage`：按照字节大小拆分为多个`storage` B 的文件
- `-l line_num`：按照行数拆分为多个行数为`line_num`的文件
- `-d`：使用数字作为输出文件名的后缀（<span class='underline_span'>无该选项时后缀为字母</span>），从0开始，不可指定  
> `--numeric-suffixes from`:使用数字作为输出文件名的后缀，`from`缺省从0开始
- `-a suffix_length`：指定生成的文件名后缀长度，`suffix_length`缺省为2
- `--verbose`：显示拆分过程的详细信息