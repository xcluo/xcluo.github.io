### 文件操作

#### 文件信息查看

- 信息查看：`ls`
- 内容统计：`wc`

#### 文件内容查看

- 原始文本：`less/more`、`head/tail` tail -f 
- 内容过滤：[`grep`](file_related/grep.md#grep)
- 内容比较：[`diff`](file_related/diff)
#### 数据格式处理

- json：[`jq`](file_related/jq.md)
- xlsx：[`xlsx2csv`]()

#### 文件修改

- 信息修改：`mv`、`cp`、`rm`、`mkdir`
- 内容连接：`cat` (文件末无\n会同行连接)
- 内容合并：`join`类似于数据库join命令、`paste`
- 内容分割：`cut`：被awk覆盖了, [`awk`](file_related/awk.md)
- 内容拆分：[`split`](file_related/split)
- 内容去重：[`sort`](file_related/sort.md) + [`uniq`](file_related/uniq.md)
- 内容替换：[`tr`](file_related/tr.md)、[`sed`](file_related/sed.md)
- 内容扰动：`shuf`
- 文件传输：[`nc`](file_related/nc.md)
- [输入输出重定向](file_related/redirection.md)：`<`、`>(>)`

### 数值操作

- format输出：[`printf`](numeric_operation/printf.md)，[`echo`](numeric_operation/echo.md)
- 序列相关：[`seq`](numeric_operation/seq.md)
### 软件安装

#### 下载工具

- `rpm`、`yum`、`apt`
- `dpkg`
- `sudo apt-get update`, `sudo apt-get upgrade`

### 进程相关
- 查看进程: [`top`](process_related/top.md)、`ps`
- 终止进程: `kill`
- watch -n 1
- [`netstat`](process_related/netstat.md)
#### 进程调度

- 多线程：[并发](process_scheduling/concurrent.md)、[并行](process_scheduling/parallel.md)

### 远程操作
- [`curl`](remote_operation/curl.md)