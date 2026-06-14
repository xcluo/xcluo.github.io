---
title: "uvicorn"
---


开发时使用uvicorn，生产应用时使用gunicorn


### uvicorn

- `uvicorn file_name:app_name` 启动服务，name无后缀名
- `unicorn file_name:app_name --reload` 自动更新修改并重新加载服务

### gunicorn

Gunicorn 负责管理多个 Uvicorn 工作进程，每个 Uvicorn 进程内部利用其异步事件循环高效处理请求。这种组合既能充分利用多核 CPU，又能享受异步带来的高并发性能

- `gunicorn main:app -w 8 -k uvicorn.workers.UvicornWorker`

Option

- `-w WORKERS` --workers，设置工作进程的数量，一个基础经验公式是(2 x CPU核心数) + 1
- `-b BIND` --bind 指定服务器绑定的地址和端口，例如 0.0.0.0:8000
- `-k WORKERCLASS` --worker-class，指定工作进程的类型。默认为 sync，可选 gevent, eventlet, gthread 等
