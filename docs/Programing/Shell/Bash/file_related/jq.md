jq：commadline JSON processor

### 安装

#### windows
1. [下载网址](https://jqlang.github.io/jq/download/)：{AMD64: 64位; i386: 32位}
2. `~/.bash_profile` 中设置 `alias jq=jqexe_absoulte_path'`
3. 加载配置 `source ~/.bash_profile`


### 使用方法
`jq [options] filter [file]`

#### `options`选项
- `-c/--compact-output`：紧凑输出，即把一个JSON对象输出在一行

#### `filter` 过滤器
- `.`：获取当前对象
- `.key`：获取对象属性
- `[]`：获取整个数组，支持切片
- `select(condition)`：过滤条件
  
```bash
# {"logits": [xxx, ...], "prob": [xxx]}

# 1. `jq`类似于 `json.loads`, `jq .prob` 得到一个字符串，仍需进一步 `jq` 处理
cat file.txt | jq .prob | jq .[]
# 2. 直接访问，推荐使用此方法
cat file.txt | jq .prob[]

# 过滤概率大于0.5的行
cat file.txt | jq select(.prob[2] > 0.5)

# 过滤概率大于0.5的行，并新增或修改name键对应的值位"luo"
cat file.txt | jq select(.prob[2] > 0.5) | jq '.name "luo"'

# 将 raw_content 与 prob 结果合并为一个数组，并过滤相应标签概率大于0.5的行，JSON对象单行输出
paste test1.txt result1.txt | awk -F '\t' '{print "[" $1 ", "$2 "]" }' | jq -c 'select(.[1].prob[1] > 0.5)'
```