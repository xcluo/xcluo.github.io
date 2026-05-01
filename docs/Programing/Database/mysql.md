

`docker run --name ai_grader-mysql -e MYSQL_ROOT_PASSWORD=你的密码 -p 3306:3306 -e MYSQL_DATABASE=exam_scoring -v mysql_data:/var/lib/mysql -d mysql:latest`

- Public Key Retrieval is not allowed
> 驱动属性中allowPublicKeyRetrieval改为true
