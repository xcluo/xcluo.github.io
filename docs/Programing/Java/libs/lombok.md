---
title: "Lombok"
---

```java
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
```

### 常用注解

#### Data

@Data 是 Lombok 库中最常用、最强大的注解之一。它是一个组合注解，相当于同时使用了

- `@ToString`: 对象的 `toString` 方法，默认为`ClassName(field1=value1, field2=value2...)`
- `@EqualsAndHashCode`: 自动生成 `equals` 和 `hashCode` （用于计算对象hash值） 方法。
- `@Getter, @Setter`: 自动生成属性的 `getter` 和 `setter` 方法，用在属性前
- `@RequiredArgsConstructor` 生成一个构造函数，参数包含所有 `final` 字段和 `@NonNull` 字段。

!!! info
    - `@ToString.Exclude` 放置在属性前，用于排除该属性出现在 `toString` 方法中。
    - `@EqualsAndHashCode.Exclude` 放置在属性前，用于排除该属性出现在 `equals` 方法和 `hashCode` 方法中。



- @AllArgsConstructor：自动生成全参构造函数。
- @NoArgsConstructor：自动生成无参构造函数。
- @EqualsAndHashCode：
