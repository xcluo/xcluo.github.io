---
title: "curl"
---

curl（**c**lient **url**）是一个功能极其强大的命令行工具和库，用于使用各种网络协议在服务器之间或与服务器传输数据。它被称为“互联网的瑞士军刀”，因为其功能丰富、用途广泛。

#### 基本语法
```shell
curl [Options] <url>
```

Options: 

- `-d escape_json` --data 发送 POST 数据  
- `-H header_json` --header 添加请求头，多个请求头需要分以`-H`开头多行表示  
- `-L` --location 自动跟随http重定向  
- `-o file_name`, --output 将输出保存为新文件  
- `-O`, --remote-name 下载并使用远程文件名保存  
- `-X request_method`, --request 指定请求方法`{GET, POST, PUT, DELETE}`  
- `-s` --silent 静默模式
- `-S` --show-error 展示error信息
- `-f` --fail 出现HTTP errors立即停止命令

#### 应用示例
```shell
curl ^
--location --request POST "https://zhenze-huhehaote.cmecloud.cn/v1/chat/completions" ^ 
--header "Authorization: Bearer 34EOY-HzL71VOtdLR0O2QbLEHUmbq_c2dbNOs1wmwvk" ^
--header "Content-Type: application/json" ^
--data-raw "{    \"model\":\"deepseek-v3\",    \"messages\": [{        \"role\": \"system\",        \"content\": \"你是哪个模型？\"    }],    \"max_tokens\": 512,\"stream\": true}"
```
> json格式数据只能单行输入