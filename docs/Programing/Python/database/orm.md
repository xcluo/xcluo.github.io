---
title: "ORM"
---

## ORM
- ORM (Object-Relational Mapping) 对象关系映射,即在python对象和关系型数据库之间建立映射关系,通过操作python对象的方式来操作数据库({`C: create, R: read, U: update, D: delete}`),而不用手动写sql(性能略低于原生sql,对于复杂查询,可能仍需要手动编写sql)
- ORM通常支持多种数据库,通过统一的接口减少数据库切换的成本
- DAO (Data Access Object)，即数据访问对象
### sqlalchemy
支持 同步/异步, 成熟稳定

```python
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import Column, DateTime, Integer, String, Text, Boolean, Float, ForeignKey
```

#### 数据表定义
```python
Column(
    unique=False,               # 是否约束为唯一值
    nullable=True,              # 是否可为空值
    index=None,                 # 是否为改列创建索引（加速排序与分组，提高查询性能）
    primary_key=False,          # 是否为主键
    autoincrement="auto",       # 是否自增
    default=0,                  # 默认值
    comment=None,               # 列注释
)

# + ForeignKey外键约束
# + relationship 实现级联操作
```

```python
class User:
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(100), nullable=False)
    full_name = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 一对多关系
    posts = relationship("Post", back_populates="author")
    orders = relationship("Order", back_populates="user")

class Post:
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    content = Column(Text)
    published = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    author_id = Column(Integer, ForeignKey("users.id"))
```


#### 数据库操作
=== "Engine"
    ```python
    db_url = f'{db_type_w_driver}://{user}:{password}@{host}:{port}/{db_path}'
    engine = create_engine(
        url=db_url,
        )
    
    # SQLite # 
    db_url = f'sqlite:///{db_path}'

    # MySQL: PyMySQL or mysqlclient #
    db_url = f'mysql+pymysql://{user}:{password}@{host}:{port}/{db_path}'
    db_url = f'mysql+mysqldb://{user}:{password}@{host}:{port}/{db_path}'

    # PostgreSQL: psycopg2 or asyncpg(异步) # 
    db_url = f'postgresql://{user}:{password}@{host}:{port}/{db_path}'
    db_url = f'postgresql{==+asyncpg==}://{user}:{password}@{host}:{port}/{db_path}'

    # SQL Server # 
    db_url = f'mssql+pyodbc://{user}:{password}@{host}:{port}/{db_path}'

    # Oracle # 
    db_url = f'oracle+cx_oracle://{user}:{password}@{host}:{port}/{db_path}'
    ```

=== "Session"
    ```python
    session_factory = sessionmaker(
        bind=engine,
        )
    
    session = Sesssion(bind=engine)
    session = session_factory()

    def get_session()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    ```

=== "CRUD"
    ```python
    # Create #
    session.add(user: User)
    session.add_all(users: List[User])
    user.posts.append(post1: Post)
    user.posts.append(post2: Post)
    session.add(user: User)             # 会自动新增关联的post1, post2
    session.commit()
    
    # Read (all, get, filter, order_by, limit, count) #
    all_users = session.query(User).all()
    user = session.query(User).get(1)
    users = session.query(User).filter(
            User.name.like("A%"),           # 名字以A开头
            User.email.contains("example")  # 邮箱包含example
        ).all()
    
    # Update #
    # 1. 查询后更新
    user = session.query(User).get(1)
    if user:
        user.name = "Alice Updated"
        user.email = "alice_updated@example.com"
        session.commit()
        print("用户更新成功")
    # 2. 批量更新
    session.query(User).update({"is_active": True})
    session.commit()

    # Delete #
    # 1. 查询后删除
    user = session.query(User).get(1)
    if user:
        session.delete(user)
        session.commit()
    # 2. 批量删除
    session.query(User).delete({"is_active": True})
    session.commit()
    ```

=== "inspect"
    ```python
    inspector = inspect(engine)
    tables = inspector.get_table_names()        #  查看数据表信息
    columns = inspector.get_columns(table_name) # 查看数据表中列信息
    df = pd.read_sql_table(table_name, engine)  # 将数据表加载成df格式
    ```

### tortoise
- `import tortoise` 异步, 轻量
- `from tortoise import fields, models`, `models.Model`

### GINO
- `GINO` 异步,轻量



### 建表相关
- sa_column：SQLAlchemy column，mysql默认65535字节≈64KB


```python
from sqlmodel import Session, SQLModel, create_engine

SQLModel.metadata

- bind=db_service.engine
- drop_all
- create_all    # 只负责 “创建不存在的表”，不负责 “更新已有表”


# 异步创建表格
async def create_db_and_tables(self):
    logger.debug("Creating database and tables (async)")

    async with self.async_engine.begin() as conn:
        try:
            await conn.run_sync(SQLModel.metadata.create_all)
            logger.debug("Tables created successfully")
        except OperationalError as oe:
            logger.warning(f"Table creation skipped due to OperationalError: {oe}")
        except Exception as exc:
            logger.error(f"Error creating tables: {exc}")
            raise RuntimeError("Error creating tables") from exc

    logger.debug('Database and tables created successfully')

field.default：python层面的默认值，创建对象时就赋值
sa_column.default：python层面的默认值，在插入表前（若未赋值）才赋值，传函数对象，不是具体值
sa_column.server_default：sqlalchemy层面的默认值

优先级：filed.default > sa.column.default > sa_column.server_default

# sa_column.server_default字段（使用text("")） 包裹
- sa.DateTime(timezone=True)： false为 timestamp，ture为timestamptz
```

| 数据类型 | PostgreSQL | MySQL| SQlite | 
| --- | --- | --- | --- |
| uuid | `gen_random_uuid()` | `UUID()` | - |
| timestamp | `CURRENT_TIMESTAMP` | `NOW()` | `CURRENT_TIMESTAMP` |
| bool | `false` | `0`（无布尔） | `0`（无布尔） |

