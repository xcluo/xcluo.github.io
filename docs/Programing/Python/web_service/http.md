---
title: "http"
---
## 请求Request
### 请求参数Params
#### 查询参数Path Params
#### 路径参数Query Params
### 请求头Headers
#### Content-Type
用于指示资源的媒体类型，告诉客户端或服务器如何解析和处理数据。
```
# 基本语法
Content-Type: media_type; charset=encoding
```

=== "应用程序类型"
    ```
    Content-Type: application/json          # json数据格式
    Content-Type: application/xml           # xml数据格式
    Content-Type: application/javascript    # js数据格式
    Content-Type: application/pdf           # pdf文件
    Content-Type: application/zip           # zip文件
    ```

=== "文本类型"
    ```
    Content-Type: text/html
    Content-Type: text/plain                # str文本
    Content-Type: text/css
    Content-Type: text/csv
    ```

=== "多媒体类型"
    ```
    Content-Type: image/jpeg
    Content-Type: image/png
    Content-Type: audio/mpeg
    Content-Type: video/mp4
    ```

=== "表单类型"
    ```
    Content-Type: application/x-www-form-urlencoded
    Content-Type: multipart/form-data
    ```
#### Authorization
用于客户端向服务器证明自己的身份
```
# 基本语法
Authorization: access_type access_token

# 示例
Authorization: Bearer 643a4348-ed5c-4858-8dce-a677a70d2632
```
### 请求体Body
### Cookies

## 响应Response