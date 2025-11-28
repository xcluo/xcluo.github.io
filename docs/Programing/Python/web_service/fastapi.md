---
title: "fastapi"
---

```bash
pip install fastapi[standard]
pip install uvicorn  
```

#### 查看网页服务
- `F12 → 网络 → 点击对应的名称，查看返回信息`

- 主要是编写接口服务，`@app.get()`一般返回json格式数据
-  使用 Pydantic 数据验证库，通过python类型提示，更方便开发效率更高
- 基于 Starlette 异步Web框架
- ASGI (Asynchronous Server Gateway Interface) Uvicorn支持高并发请求
- 内置Swagger UI(docs)和ReDoc(redoc)，自动是生成交互式API文档
- FastAPI() 是应用的主体，而 APIRouter() 是应用的模块化组件
- 起服务
    - `uvicorn file_name:app_name --reload`
    - `fastapi dev file_name.py` 调试fastapi代码
    - `python file_name.py`
        ```python
        uvicrrn.run(
            app,
            *,
            host="127.0.0.1",
            port=8000,
            reload=False,           # 是否自动更新改动
            log_level=None,
        )
        ```



1. get
```python
# url重复定义或定义冲突时，会按照声明顺序贪心匹配路径，路径参数匹配时即调用

# 路径参数:传入的是什么space_name, url则用相应space_name渲染; 通过requests中的url参数传入
@app.get("/knowledge/{space_name}")
async def batch_document_sync(
    space_name: str,
)



# 查询参数:在url后 + `?参数名=参数值&参数名=参数值`
@app.get("/lxc/")
def test_query(a, b):
    return {"hello": a, "world123": b}


class LXC(BaseModel):
    name: str
    age: int
    goods: list[str] = []

@app.get("/", response_model=LXC)
def lxc():
    # return LXC(name="xcluo", age=18)
    return {"name": "xcluo", "age": 18, "goods": []}
# - reponse_model限定返回的数据类型, 返回为json时会自动转型为BaseModel格式(要求json数据字段集大于等于BaseModel字段集合),因此两种方式均可
# - response_model_exclude_unset, 不返回未设值的字段(即不返回未设值的字段), 本质上与dict兼容
# - response_model_exclude_defaults
# - response_model_exclude_none
```

#### 请求体
`from fastapi import Body`
```python
requests.post(
    url,            # 通过指定目标url调用相应的请求体函数
    json=None,      # 传入的json格式数据
)

```
- `from enum import Enum`
- 传递方式:直接访问页面;通过本地代码访问
- `from fastapi import Query`,查询参数验证
    - `...` 该查询参数必须提供
    - `gt, ge, lt, le` 大于，大于等于，小于，小于等于
    - `min_length, max_length` 最小长度，最大长度
    - `alais` 设置前端接口调用别名,缺省为变量名
    - `regex, pattern` 通过正则表达式限制查询参数格式
    - `description` 查询参数的描述信息
    - `example, examples` 查询参数的示例
    - `deprecated` 该查询参数是否已被弃用
  
- `from fastapi import Paht`, 路径参数验证
    - `...` 该路径参数必须提供
    - `gt, ge, lt, le` 大于，大于等于，小于，小于等于
    - `min_length, max_length` 最小长度，最大长度
    - `alais` 设置前端接口调用别名,缺省为变量名
    - `regex, pattern` 通过正则表达式限制路径参数格式
    - `description` 路径参数的描述信息
    - `example, examples` 路径参数的示例
    - `deprecated` 该路径参数是否已被弃用

- 自定义验证规则:`typing + pydantic`
- `from pydantic import Field` 搭配BaseModel进行参数验证
- 表单数据类型:`from fastapi import Form` `user_name: str = Form(...)`, 引发`-H application/x-www-form-urlencoded` 而不是`application/json`, 查询参数此时不会出现在url处而是在数据 `-d` 处

#### 异步处理
- `async def`协程coroutines,进一步对协程并发(IO和cpu)
- `await` 放出异步任务占用,等待任务完成，再执行后续任务

#### 文件上传
- `pip install python-multipart`,`from fastapi import File, UploadFile` 上传文件
- `file: bytes=File(...)`,上传表单数据,docs中会自动出现选择文件button, 文件数据内容会存在`-F`中
- `file: UploadFile` 大文件上传(默认异步), 可通过`file.filename`获取上传文件名,常需搭配`async def ... await` 使用
- `import aiofiles` 异步打开文件 `async witha iofiles.open()`, 时间戳 (14, 18:00)
- `import asyncio`

- 上传多个文件`files: List[UploadFile] = File(...)`
#### requests
- 前部参数：通过路径渲染传递
- 中部参数：通过请求体传递
- 后部参数：可通过Depends依赖注入确定
- `from fastapi import HTTPException` 返回
- 服务器响应码,`{1: 信息响应,已接收,正在处理; 2: 成功响应; 3: 重定向; 4: client问题; 5: server问题}`
- `from fastapi import Request` 获取用户的request请求对象, 用于操控请求信息


```python
# FastAPI和APIRouter均能进行include_router # 
app.include_router(
    router,
    prefix="",      # 为router设置的url前缀, 即 ip:port/suffix/path, 一般在各APIRouter中设置较为方便
    tags=[],        # 路由标签注释
)
```


```python
class Result(           
    BaseModel,          # 继承 Pydantic 基类
    Generic[T]          # 声明为泛型类，T 是类型参数
    success: bool
    err_code: Optional[str] = None
    err_msg: Optional[str] = None
    data: Optional[T] = None

    @classmethod
    def succ(cls, data: T):
        return Result(success=True, err_code=None, err_msg=None, data=data)
)
Result[list]        # 指定 list 为类型参数
Result.succ("lxc")  # 不指定类型参数直接声明,等价于Result[str].succ("lxc")
```

### 响应类型response
#### 文件
- 客户端`requests.post(stream=True)` 表示流式读取
```bash
# StreamingResponse除了用于流媒体输出，也可用于AR型LLM的时许输出
from fastapi import Response, FileResponse, StreamingResponse

app = FastAPI()
@app.get("/custom-file"):
async def get_custion_file():
    info = b"file content"
    return Response(
        content=info,
        media_type="text/plain",    # {text/pain: 文本内容, application/pdf: pdf文件, }
        headers={"Content-Disposition": "attachment;filename='file.txt'"}
                                    # attachment表示立即下载
    )
    return Response(
        path=file_path,
        media_type="application/pdf",    # {text/pain: 文本内容, application/pdf: pdf文件, vedio/mp4: mp4文件}
        headers={"Content-Disposition": "attachment;filename='file.pdf'"}
                                    # attachment表示立即下载, filename表示默认文件名
    )
```
- `from fastapi.responses import HTMLResponse` 响应格式控制,将返回的结果按照什么格式渲染解析

- `from fastapi.staticfiles import StaticFiles`
- `app.mount`: 为网页地址挂载html前端

#### 中间件
- `@app.middleware`: 中间件, 在client和server中间, `client ⟷ middleware ⟷ server`, 中间件是当前api所有接口通用的,且都要执行
- **中间件调用顺序**:根据中间件声明顺序即`client`至`server`流程中的调用顺序,类比于递归调用理解
- `app.add_middleware()` 自定义中间件
![alt text](image.png)
- `app = request_middleware(app)` 一定需要赋值么？
### 资源共享
- SOP(same-origin policy),源=[协议(http/https) + 域名 + 端口号],所有元素一致时才为同源
- CORS(cross-origin resource sharing),跨域资源共享, 自定义某些白名单源可以访问服务器资源,从而绕过同源策略

### APIRouter 业务划分
- `from fastapi import APIRouter`

### 依赖注入
- Dependence Injection (DI)
- 依赖注入`from fastapi import Depends` 在参数传递至目标函数前，先进行前置注入函数的处理
#### 路径级
- `@app.get(dependencies=[Depends(path_di_func)])` 函数注入
- 对于含传入参数的函数，可对传入各参数分别进行Depends注入
#### 路由级
影响当前apirouter下的所有路由
- `APIRouter(dependencies=[Depends(router_di_func)])`
- 注入函数的传入参数为`token: str = Header(...)`
#### 全局级
- `FastAPI(dependencies=[Depends(global_di_func)])`
- 注入函数的传入参数为`request: Request`