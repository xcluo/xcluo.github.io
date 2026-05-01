---
title: "async operation"
---

I/O密集型操作使用异步，CPU密集型操作考虑使用线程池


### async def + await

### asyncio

- 同步函数转异步

```python
# 同步 - 异步桥接工具，将同步函数提交到线程池，转为异步任务
# 仅能异步函数中执行同步函数，管同步兼容
# func内里实际调用的是同步函数：loop.run_in_executor 直接返回结果
# func内里实际调用的是异步函数且return：loop.run_in_executor 返回协程对象
# func内里实际调用的是异步函数且yield：loop.run_in_executor 返回async generator
loop = asyncio.get_event_loop()

init_response = await loop.run_in_executor(None, sync_func, *args)
```

- `asyncio.gather` 仅能异步函数中并发批量执行异步任务，结果顺序与传入任务的顺序一致，比单纯await更快
- `asyncio.as_completed` 批量并发执行异步任务，结果不保序
- `asyncio.create_task` 生成异步并发任务，搭配await 几乎等价于gather
- `if asyncio.iscoroutine(init_response): ret = await init_response` 如果是协程，则返回协程结果
- `asyncio.to_thread`
- `asyncio.run` 任意（同/异步函数）场景启动异步（函数）事件循环
- `asyncio.ensure_future`
- `asyncio.wait_for`

- `asyncio.Semaphore` 异步信号量，用于并发控制
- `asyncio.as_completed(tasks)` 批量并发执行异步任务，结果不保序
    - 此时可以在构建任务时使用dict
    - as_completed时传入keys()
    - 处理输出结果时，使用dict.get(key)获取源信息

### aiohttp

- `Session → session.post`
- `aiohttp.ClientSession` 创建aiohttp会话
- `async_session.post` 发起异步POST请求

```python
async def main():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json=payload,
            headers=headers
        ) as res:
            ...  # 具体任务逻辑，常通过yield分段返回
```

### aiofiles

异步读取文件

```python
async def upload_file(file: UploadFile = File(...)):
    # 异步打开文件 async + aiofiles
    async with aiofiles.open(f"uploads/{file.filename}", "wb") as f:
        # 异步读取传入的流信息
        content = await file.read()
        # 异步写入文件
        await f.write(content)
    
    return {"filename": file.filename, "status": "uploaded"}
```

