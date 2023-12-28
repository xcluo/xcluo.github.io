sed (**s**tream **ed**iter for filtering and transforming text) 是以为单位处理文本数据，可以对数据进行过滤、替换、删除和插入等功能。

### 安装
#### windows
1. [下载网址](https://sourceforge.net/projects/gnuwin32/files/sed/)
2. 选择版本的对应的 `setup.exe` 进行安装即可

### sed
`sed [OPTION] ... {script-only-if-no-other-script} [input-file] ...`

#### `OPTION`
- `-n/--quiet/--silent`：取消默认打印所有内容，而是只打印经过sed处理后的行
- `-e script/--expression=script`，执行的脚本，多个 `-e` 表示非短路逻辑或，可借助 `uniq` 实现输出去重
- `-f script-file/--file=script-file`，执行文件内的脚本，和 `-e` 区别在于该方式脚本存放在已有文件
- `-i[SUFFIX]/--in-place[=SUFFIX]`：直接修改读取的文件内容，而不是输出到终端

#### `script`
格式 `[address[,address]]s/pattern-find/replacement-pattern/[g,p,w,n]`
```bash
# 前/后插入 i, a # 
sed 'a lxc'             # 在每一行 后 插入 lxc，同理a操作可以改为i操作
sed '2a lxc'            # 从第2行开始在每一行 后 插入 lxc，同理a操作可以改为i操作 
sed '$a lxc'            # 只在最后一行 后 插入 lxc，同理a操作可以改为i操作
sed '/a/a lxc'          # 在含有字符 a 的行后插入 lxc，同理a操作可以改为i操作

# 删除 d #
sed '2,10d'             # 删除 [2, 10) 行
sed '2,10!d'            # !取反，即删除 all - [2, 10) = [1, 2) + [10, len] 行
sed '2,10{/^$/d}'       # 删除[2, 10) 行内的空行
                        # 由于 {} 是特殊符号，因此正则表达式时需要转义
sed '{/^$/d}'           # 删除所有空行

# 修改 s #
sed '2,5s/a/8/'         # [2, 5) 行内的 a 修改为 8

# 替换 c #
sed '2c lxc'            # 第2行内容替换为 lxc
sed '2,$c lxc'          # 第2至最后一行内容替换为 lxc

# 查看 p, = #
sed -n 's/a/8/p'        # 查看所有 a 改为 8 的行
sed -n '/a/='           # 查看所有含字符 a 的行号
sed -n '/^[A-Z]\{3\}/p' # 查看所有以3个大写字母开头的行
                        # \{ \} 需要转义
```
> `!` 表示条件取反，用于 command 前  
> `;` 表示不短路的条件或，可借助 `uniq` 命令实现去重  
> `{` 表示原始字符，`\{` 表示正则中表数量的转义字符