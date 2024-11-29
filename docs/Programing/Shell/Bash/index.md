### 文件操作

#### 文件信息查看

- 信息查看：`ls`
- 内容统计：`wc`

#### 文件内容查看

- 原始文本：`less/more`、`head/tail`
- 条件选择：[`awk`](file_related/awk.md)
- 内容过滤：[`grep`](file_related/grep/#grep)
- 内容比较：[`diff`](file_related/diff)

#### 数据格式处理

- json：[`jq`](file_related/jq)
- xlsx：[`xlsx2csv`]()

#### 文件修改

- 信息修改：`mv`、`cp`、`rm`
- 内容连接：`cat` (文件末无\n会同行连接)
- 内容合并：`join`、`paste`
- 内容分割：`cut`
- 内容拆分：[`split`](file_related/split)
- 内容去重：`sort` + `uniq`
- 内容替换：`tr`、[`sed`](file_related/sed)
- 内容扰动：`shuf`
- 文件传输：[`nc`](file_related/nc)
- [输入输出重定向](file_related/redirection)：`<`、`>(>)`

### 数值操作

- format输出：[`printf`](numeric_operation/printf.md)

### 软件安装

#### 下载工具

- `rpm`、`yum`、`apt`

### 进程相关
- 查看进程: [`top`](process_related/top)、`ps`
- 终止进程: `kill`

#### 进程调度

- 多线程：[并发](process_scheduling/concurrent.md)、[并行](process_scheduling/parallel.md)