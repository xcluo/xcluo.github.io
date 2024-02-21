```bash title="json内容去重"
# 1. 头尾分别插入[ 和 ]，每行插入分割符，形成一个list
# 2. unique_by只在list中生效
cat <file_name> | sed -e '1i[' -e '2,$i ,' -e '$a]' | jq -c 'unique_by(.<key_name>) | .[]'  > unique_result.json
```