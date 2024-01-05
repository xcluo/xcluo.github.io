[jq](https://jqlang.github.io/jq/manual/)：commadline JSON processor

### 安装

#### windows
1. [下载网址](https://jqlang.github.io/jq/download/)：{AMD64: 64位; i386: 32位}
2. `~/.bash_profile` 中设置 `alias jq=jqexe_absoulte_path'`
3. 加载配置 `source ~/.bash_profile`


### 使用方法
`jq [options] filter [file]`

#### `options`选项
- `-c/--compact-output`：紧凑输出，即把一个JSON对象输出在一行
> 通过紧凑输出处理的行中，取消了键与值之间的空格，即 `"key":"value"`
#### `filter` 过滤器
- `.`：获取当前对象
- `.key`：获取对象属性
- `[]`：获取整个数组，支持切片
- `select(condition)`：过滤条件
  
```bash
# {"c", "xxx”, "logits": [xxx, ...], "prob": [xxx]}

# 1. `jq`类似于 `json.loads`, `jq .prob` 得到一个字符串，仍需进一步 `jq` 处理
jq .prob | jq .[]
# 2. 直接访问，推荐使用此方法
jq .prob[]

# 数据重组
jq '[.a, .b]'
jq '{"content": .c}'

# 过滤概率大于0.5的行
jq select(.prob[2] > 0.5)

# 过滤概率大于0.5的行，并新增或修改name键对应的值位"luo"
jq select(.prob[2] > 0.5) | jq '.name "luo"'

# 过滤内容c长度大于200的行 （管道符不限制段数）
jq -c 'select(.c | length > 200)'        # 保留整个行
jq -c '.c | select(length > 200)'        # 仅保留字段c对应的值
jq -c 'select(.c | length > 200) | .c'   # 等价于↑


# 通过`paste`和`awk`命令将 raw_content 与 prob 结果合并为一个数组
# + 过滤相应标签概率大于0.5的行
# + 最终JSON对象单行输出
paste test1.txt result1.txt | awk -F '\t' '{print "[" $1 ", "$2 "]" }' | jq -c 'select(.[1].prob[1] > 0.5)'

# [{"c": "xxx", "l": "x"}, {"c": "xxx", "l": "x"}]

# unique_by(.file_name) 适用于单个json list的关键字去重
# 去重 + 并选择内容长度处于 (0, 200] 区间的样本
jq -c 'unique_by(.c) | .[] | select(.c | lenght > 0 and length <= 200)'

# 对键值c去重
sed -e '1i[' -e '2,$i ,' -e '$a]' | jq -c 'unique_by(.c) | .[]'
```