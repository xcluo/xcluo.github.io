---
title: nginx
---

- Nginx 是一个 应用程序（Web 服务器、反向代理、负载均衡器）。
- 单机 Nginx 只能管理一台机器上的容器，机器挂了就全挂了。
- K8s + Nginx Ingress 可以让 Nginx 以多副本形式运行在多台机器上，前端再挂一个云负载均衡器，实现了 高可用入口。并且 Nginx 的配置可以随着服务的扩缩容自动更新（比如后端 Pod 新增了 IP，Nginx 会自动感知并加入 upstream），无需人工干预。

- [下载网址](https://nginx.org/en/download.html)
- 假如环境变量：nginx.exe所在的目录
- 启动nginx：进入nginx.exe所在的目录，执行nginx命令

- nginx中service端口表示nginx访问业务服务的端口
- dockerfile中的端口表示容器和宿主机映射端口

### nginx.conf

```nginx

```
