---
title: "requests"
---

### 请求方式

- requests.get
- requests.post
- requests.put
- requests.delete
- requests.patch
- requests.head
- requests.options
- requests.trace

### 请求参数

- url
- data：用于传输表单数据，此时默认 `Content-Type: application/x-www-form-urlencoded`
- json：此时默认`Content-Type: application/json`

- stream：`{++False++: 下载整个响应内容; True: 允许逐块处理响应内容}`

    ```python
    # for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
    for chunk in response.iter_lines():
        print(chunk)
    ```

- files：
    
    - 上传多个文件：`[(file_alias, (file_name, file_obj, content_type)), ...]`
    - 上传单个文件：`{file_alias: (file_name, file_obj, content_type)}`, `{file_alias: file_obj}`
    - `file_obj = open(file_path, mode='rb')`

- params：用于传递查询参数，也可直接通过&连接`key=value`方式传递；路径参数直接输入渲染后的url即可，此时路径参数已在url中
- headers
- timeout
- cookies
- allorw_redirect