---
title: "SpringBoot"
---

```java
import org.springframework.beans.factory.annotation.Value;
import org.springframework.beans.factory.annotation.Autowired;
```

### 注解

#### Value

- @Value("${risk.level.low}") 从环境文件中读取
- @Value("hello world") 直接赋值

#### Autowired

- 保证字段不为 null（对象创建时必须传入）。“Spring 老大，我这儿缺一个能操作‘重点人群’数据的工具（Mapper），你帮我在你的工具箱里找一个现成的、配置好的给我装上，我直接用就行，不用我自己造。”


### Restful

- `@RestController`
  `@RequestMapping` 映射URL路径，可用于类或方法。类上定义基础路径，方法上定义具体路径
- `@GetMapping` 映射 GET 请求
- `@PostMapping` 映射 POST 请求
- `@PutMapping` 映射 PUT 请求
- `@DeleteMapping` 映射 DELETE 请求
- `@RequestParam`
- `@PathVariable` 路径参数
- `@RequestBody`

### bean
- `@Service` 标注业务层组件，Spring会自动扫描并注册为Bean
- `@Component` 标注通用组件，Spring会自动扫描
- `@Repository` 标注数据访问层组件，Spring会自动注册，同时开启异常翻译
