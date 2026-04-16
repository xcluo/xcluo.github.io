

### chmod

change mode，修改文件或目录的权限（执行，读，写等），语法格式如下 `chmod [OPTIONS] MODE[,MODE] FILE ...`

OPTION

- `-R` --recursive 递归修改文件和文件夹
- `-V` --verbose 显示处理过程

MODE，符号模式`[ugoa...][[+-=][rwxXst...]...]`

- `{u: 文件所有者, g: 组, o: 其他, a: 所有用户}`
- `{+: 添加权限, -: 移除权限, =: 设置精准权限}`
- `{r: 可读, w: 可写, x: 可执行}`

MODE，数字模式（3位八进制，分别表示`rwx`）

### chown

change owner，修改文件或目录的所有者和所属组，语法格式如下 `chown [OPTIONS]... [OWNER][:[GROUP]] FILE...`

OPTION

- `-R` --recursive，递归修改文件和文件夹
- `-v` --verbose，显示详细处理过程
- `-c` --changes，显示修改之处的详细处理过程

OWNER：用户名或用户id，将文件所属用户设为OWNER
GROUP：组名或组id，将文件所属组设为GROUP

!!! info
    不要求USER是否属于GROUP
