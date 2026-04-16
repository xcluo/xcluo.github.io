---
title: "Daytona"
---
- [Daytona Docs](https://www.daytona.io/docs/en/)
- 常用Python和TypeScript调用

### Image/Snapshot
- image命名格式 `container_registry/name:tag`，不支持latestd等不明确版本的标签
- image初始化snapshot，snapshot可以保存为image（类似于初始化模型和ckpt的关系）
- image entrypoint：修改启动逻辑后再创建snapshot
    - `/usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf`
    - `/usr/bin/supervisord` 为可执行程序绝对路径
    - `-n` 非守护进程模式 no daemon 启动
    - `-c` 指定配置文件路径（configuration file）
    - `/etc/supervisor/conf.d/supervisord.conf` 配置文件绝对路径

```python
daytona_config = DaytonaConfig(
    api_key=DAYTONA_API_KEY,
    api_url=DAYTONA_SERVER_URL,     # `server_url` 等价于 `api_url`
    target: Optional[str] = None    # Organization Region, {"us", "eu"}
)
daytona = Daytona(daytona_config)
```


### Sandbox
- 无活动15分钟自动停止, 停止7天自动存档
- 默认1 v(irtual)CPU, 1GB RAM, and 3GiB disk.
- 状态：STARTED, STOPPED, DELETED, ARCHIVED(存档)
- To keep the Sandbox running indefinitely without interruption, set the auto-stop value to 0 during creation.

```python
params = CreateSandboxParams(
    language: Optional[CodeLanguage] = None,    # {python, javascript, typescript}，默认为python
    image: Optional[str] = None,                # 指定镜像
    name: Optional[str] = None,                 # 默认为sandbox_id
    resources={
        "cpu": 2,       # cpu核数
        "memory": 4,    # RAM大小，GB
        "disk": 5,      # disk大小，GB
    }
)
# 根据id或name访存sandbox
sandbox = daytona.get/find_one(sandbox_id_or_name)
sandbox = daytona.create(
    params: Optional[Union[CreateSandboxFromSnapshotParams, CreateSandboxFromImageParams]] = None,  # 默认从新创建镜像，再创建sandbox
)
sandbox = daytona.create(

)

sandbox.id

sandbox.stop()
sandbox.start()
sandbox.archive()
sandbox.delete()
```
#### code command
```python
response = sandbox.process.exec(
    code: str,
    params: Optional[CodeRunParams] = None, # argv: Optional[List[str]] = None
                                            # env: Optional[Dict[str, str]] = None

    timeout: Optional[int] = None,
)

response.exit_code
response.result
```

#### bash command
```python
# Execute a bash command
response = sandbox.process.exec("echo 'Hello, World!'")
```


- daytona.get_current_sandbox(sandbox_id)
- sandbox.get_preview_link 生成当前沙箱环境中运行的应用 / 服务的临时可访问预览URL
- sandbox.fs

