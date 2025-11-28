---
title: "async operation"
---

I/O密集型操作使用异步，CPU密集型操作考虑使用线程池


### async def + await

### asyncio

### aiofiles
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

### aiohttp