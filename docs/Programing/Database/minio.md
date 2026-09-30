---
title: "MinIO"
---

`docker exec -it CONTAINER printenv | grep MINIO`，进入容器并打印所有MINIO相关的环境变量

新版本：MinIO从v5版本起，统一使用 MINIO_ROOT_USER 和 MINIO_ROOT_PASSWORD 作为根用户凭证。MINIO_ROOT_USER 对应 Access_Key，MINIO_ROOT_PASSWORD 对应 Secret_Key。

旧版本：如果你使用的是较旧的镜像，变量名可能是 MINIO_ACCESS_KEY 和 MINIO_SECRET_KEY，不过它们已被弃用。
