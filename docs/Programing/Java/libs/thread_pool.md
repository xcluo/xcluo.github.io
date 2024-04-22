#### Executorservice

```java
ExecutorService executor = Executors.newFixedThreadPool(poolSize);

// 提交多个任务给线程池
for (int i = 0; i < poolSize; i++) {
    // todo: 注意i是局部变量，匿名函数中访问不到，需在 {} 域内再次声明
    executor.submit(() -> {
        // 自定义的方法
    });
}

// 关闭线程池，防止结束后不关闭现象
executor.shutdown();
```
!!! info
    该方法只能在main函数中运行