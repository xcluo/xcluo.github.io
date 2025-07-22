`awk`：其名称得自于它的创始人阿尔佛雷德·**艾**侯、彼得·**温**伯格和布莱恩·**柯**林汉姓氏的首个字母

#### 语法格式
`awk [options] '条件 {动作} 条件 {动作} ...' file_1 file_2 ...`
> `条件 {动作}` 部分是字符串，其内容无转义字符`$0、$n`时单引号可以用双引号来进行形参替换

### 变量
#### 内置变量
| 变量名 | 描述 |
| --- | --- |
| `FILENAME` | 当前输入文档的名称 |
| `FNR` | File Number of the Current Record，当前输入文档的当前行号，当有多个输入文档时，`FNR`的值会从1重新开始 |
| `NR` | Number of the Current Record，输入数据流的当前行号，当有多个输入文档时会持续计数，不从1开始 |
| `$0` | 当前行的全部数据内容，需用单引号表示以防止转义，`'{print $0}'` |
| `$n...` | 当前行的第$n$个字段的内容($n\ge 1$)，，需用单引号表示以防止转义，`'{print $1， $2, $3}'` |
| `NF` | Number of Fileds，当前记录（行）的字段（列）个数 |
| `FS` | Field Separator，字段分隔符，默认为空格或者Tab制表符 |
| `OFS` | Output Field Separator，输出字段（列）分隔符，默认为空格 |
| `ORS` | Output Record Separator，输出记录（行）分隔符，默认为换行符\n |
| `RS` | Record Separator，设置记录（行）分隔符，默认为换行符\n |  

>- 记录Record表示行，字段Field表示列，Field之间用`FS`分隔，Record之间用`RS`分隔  
>- NR将所有文件视作一个数据流，NFR将各个文件分别视作一个数据流

#### 自定义变量


### 常用awk命令
```bash
awk "NR==6" file_name               # 显示第6行内容
awk “NR<=6” file_name               # 显示前6行内容
awk "NR>4 && NR<11" file_name       # 显示5-10行
awk "NR%2==0" file_name             # 显示奇数行

awk -F ' '  '{print $1}' file_name  # 显示由 ' ' 分割的第一项（项下标由1开始）
                                    # -F ' '可简写为 -F' '
awk '$1 > .5 {count++} END {print count}' file_name
                                    # 统计数值行大于0.5的行数
```