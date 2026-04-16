---
title: Redis
---
Redis（Remote Dictionary Server ），即远程字典服务

```python
import redis                    # 同步操作
import redis.asyncio as redis   # 异步操作
```


#### 客户端操作
=== "ping"
    检测 Redis 客户端与服务端连接状态
    ```python
    client.ping(
        **kwargs
    )
    ```

=== "info"
    获取 Redis 服务器的各类运行状态信息和统计数据
    ```python
    client.info(
        section: Union[str, None] = None,   # 指定分区
    )
    ```

#### 基础CRUD

=== "set"
    ```python
    client.set(
        name: KeyT,
        value: EncodableT,
        ex: Union[ExpiryT, None] = None,        # 过期时间 seconds
        px: Union[ExpiryT, None] = None,        # 过期时间 milliseconds
        nx: bool = False,
        xx: bool = False,
        keepttl: bool = False,
        get: bool = False,
        exat: Union[AbsExpiryT, None] = None,
        pxat: Union[AbsExpiryT, None] = None,
    )
    ```

=== "exists"
    判断传入的所有键存在于数据库中的数量（有效键的个数）
    ```python
    client.exists(
        *names: KeyT
    )
    ```

=== "keys"
    以列表形式返回匹配给定模式的所有键
    ```python
    client.keys(
        pattern: PatternT = "*", 
        **kwargs
    )
    ```

=== "get"
    判断键是否存在，存在返回对应值，不存在返回None
    ```python
    client.get(
        name: KeyT
    )
    ```

=== "delete"
    删除一个或多个指定的输入键及其对应的值，并返回成功删除的键的数量
    ```python
    client.delete(
        *names: KeyT
    )
    ```


#### 列表数据操作

=== "增改操作"
    - **`l/rpush(x)`**：向指定的 key 对应列表（List）的头/尾部（左/右侧）追加一个或多个元素
    - **`lset`**：修改指定的 key 对应列表中指定索引的元素
    - **`linsert`**：在指定的 key 对应列表中指定元素的指定方向插入元素
    ```python
    # 若key不存在，则会先创建再追加（可用于创建列表），返回最终列表长度
    client.l/rpush(
        name: str, 
        *values: FieldT
    )

    # 如key不存在，不追加，返回0
    client.lpushx/rpushx(
        name: str, 
        *values: FieldT
    )

    client.lset(
        name: str, 
        index: int, 
        value: str
    )

    client.linsert(
        name: str, 
        where: str,     # {"before", "after"}
        refvalue: str,  # 指定元素（需存在于列表中）
        value: str
    )
    ```

=== "删除操作"
    - **`(b)l/rpop`**：删除指定键对应列表的左/右侧元素，并返回pop出的元素
    - **`lrem`**：删除指定键对应列表中删除指定键对应列表中值为value的元素
    - **`ltrim`**：保留指定键对应列表中指定范围[start, end]的元素
    ```python
    client.l/rpop(
        name: str,
        count: Optional[int] = None,
    )

    # 当列表为空时，会阻塞客户端，直到有元素添加或超时退出（适用于阻塞队列场景）
    client.bl/brpop(
        keys: List, 
        timeout: Optional[int] = 0
    )

    client.lrem(
        name: str, 
        count: int,     # count=0，删除所有值为value的元素
                        # count>0，删除→前count个值为value的元素
                        # count<0，删除←后count个值为value的元素
        value: str
    )

    client.ltrim(
        name: str, 
        start: int, 
        end: int
    )
    ```

=== "查询操作"
    ```python
    # 获取指定键对应列表的长度
    client.llen(
         name: str
    )

    # 全称 List Range，查询指定 key 对应列表切片[start, end]
    client.lrange(
        name: str, 
        start: int, 
        end: int
    )
    
    # 获取指定键对应列表中指定索引对应的元素
    client.lindex(
        name: str, 
        index: int
    )
    ```
#### 流式数据操作
```
# 全称 Stream Range，按照消息 ID 的有序范围，从 Redis Stream 中获取对应的消息子集，且返回的消息始终按消息 ID 从小到大（时间递增）排序。
client.xrange(
    name: KeyT,
    min: StreamIdT = "-",
    max: StreamIdT = "+",
    count: Union[int, None] = None,
)
client.xadd(
    name: KeyT,
    fields: Dict[FieldT, EncodableT],
    id: StreamIdT = "*",
    maxlen: Union[int, None] = None,
    approximate: bool = True,
    nomkstream: bool = False,
    minid: Union[StreamIdT, None] = None,
    limit: Union[int, None] = None,
)
client.xread(
    streams: Dict[KeyT, StreamIdT],
    count: Union[int, None] = None,
    block: Union[int, None] = None,
)
client.xlen(
    name: KeyT
)
client.xtrim(
    name: KeyT,
    maxlen: Union[int, None] = None,
    approximate: bool = True,
    minid: Union[StreamIdT, None] = None,
    limit: Union[int, None] = None,
)
```
```
# name对应值增加amount，若键不存在，值则为amount
client.incr(
    name: KeyT, 
    amount: int = 1
)
# name对应值减少amount，若键不存在，值则为-amount
client.decr(
    name: KeyT, 
    amount: int = 1
)
```

#### 广播操作
=== "发布者"
    向 Redis 中指定的频道发布一条消息，所有已订阅该频道的客户端都会实时接收到这条消息
    > 发布的消息不会被 Redis 存储，若此时无订阅者，消息直接丢失

    ```python
    client.publish(
        channel: ChannelT,      # 频道名
        message: EncodableT,    # 待传递的消息
        **kwargs
    )
    ```

=== "订阅者"
    - **`client.pubsub`** 创建广播订阅对象
    - **`(p)subscribe`** 订阅一个或多个频道
    - **`(p)unsubscribe`** 取消订阅一个或多个频道
    - **`listen`** 阻塞式监听广播消息（返回迭代器，持续等待消息，推荐生产环境使用）
    - **`close`** 关闭广播订阅对象
    ```python
    pubsub = client.pubsub(
        **kwargs
    )

    pubsub.subscribe(
        *args,                  # *channels
        **kwargs
    )
    # pattern subscribe
    pubsub.psubscribe(
        *args,                  # *channel patterns
        **kwargs
    )
    pubsub.unsubscribe(
        *args                   # *channels
    )

    # pattern unsubscribe
    pubsub.punsubscribe(
        *args                   # *channel patterns
    )

    pubsub.listen()

    pubsub.close()
    ```

### CLI命令
为便于区分，命令均使用大写形式（本质上可小写）

#### 远程登录
基本语法 `redis-cli [OPTIONS]`

Options:

- `-h host_name` 主机ip地址，默认127.0.0.1
- `-p port` 端口号，默认6379
- `-a password` 登录密码
- `-n db_num` 数据库号码

#### CRUD命令

=== "Create"
	```bash
	SET key value [EX time_ex] [NX time_nx] [XX time_xx]
	MSET key1 value1 key2 value2 ...
	SETEX key time_ex value
	```
	
=== "Read"
	```bash
	GET key
	MGET key1 key2 ...
	```

=== "Update"
	```bash
	INCR key
	DECR key
	INCRBY key step
	```
=== "Delete"
	```bash
	DEL key
	```

#### 列表命令
- LPUSH list_key value1 value2 ...
- RPUSH list_key value1 value2 ...
- LPOP list_key
- RPOP list_key
- LRANGE list-key start_idx end_idx		# [start_idx: end_idx]
- LLEN list_key
- LREM list_key count_num value