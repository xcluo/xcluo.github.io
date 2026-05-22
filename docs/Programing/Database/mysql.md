
### 启动容器
`docker run --name ai_grader-mysql -e MYSQL_ROOT_PASSWORD=你的密码 -p 3306:3306 -e MYSQL_DATABASE=exam_scoring -v mysql_data:/var/lib/mysql -d mysql:latest`

- Public Key Retrieval is not allowed
> 驱动属性中allowPublicKeyRetrieval改为true

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