
### 启动容器

`docker run --name ai_grader-mysql -e MYSQL_ROOT_PASSWORD=你的密码 -p 3306:3306 -e MYSQL_DATABASE=exam_scoring -v mysql_data:/var/lib/mysql -d mysql:latest`

- Public Key Retrieval is not allowed

> 驱动属性中allowPublicKeyRetrieval改为true

### MySQL 服务器操作

#### 连接服务器

`mysql -u root -p`

- `-u` 指定用户名
- `-p` 提示输入密码（密码与 `-p` 之间无空格）
- 连接成功后进入 MySQL 交互界面，提示符变为 `mysql>` 后输入密码

#### 数据库操作

```sql
-- 显示 MySQL 服务器中所有可用的数据库
SHOW DATABASES;

-- 选择数据库，之后所有的 操作都在该数据库下进行
USE database_name;
```

#### 数据表操作

```sql
-- 显示当前数据库中所有的数据表
SHOW TABLES;
```

```sql
-- 查看指定数据表的结构（列名、数据类型、约束等）
DESC table_name;
DESCRIBE table_name;
```

#### 数据表中新增列

```sql
ALTER TABLE table_name ADD COLUMN column_name 数据类型 [约束];
```

- 示例：在 `students` 表中新增 `email` 列

```sql
ALTER TABLE students ADD COLUMN email VARCHAR(100);
```

- 若要在指定位置新增列，可使用 `AFTER` 或 `FIRST`

```sql
-- 在指定列之后新增
ALTER TABLE 表名 ADD COLUMN 新列名 数据类型 AFTER 已有列名;

-- 在第一列新增
ALTER TABLE 表名 ADD COLUMN 新列名 数据类型 FIRST;
```

---

### 数据迁移

#### 数据导出

`docker exec -i CONTAINER mysqldump -uroot -pvr-test --all-databases > backup.sql`

- `-i` 保持标准输出打开
- `CONTAINER` 容器号
- `mysqldump -u{user_name} -p{password}` 导出mysql数据库，账号名和密码与`-u`和`-p`间不存在空格
- `--all-databases` 导出所有数据库
- `> backup.sql` 将导出结果保存为宿主机中文件

#### 数据导入

`docker exec -i CONTAINER mysql -uroot -pvr-test < backup.sql`

- `mysql -u{user_name} -p{password}` 导入mysql数据库，账号名和密码与`-u`和`-p`间不存在空格
- `< backup.sql` 将宿主机文件作为输入信息