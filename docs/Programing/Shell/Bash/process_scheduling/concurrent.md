```bash
#!/bin/bash


# 设置运行次数或与运行对象
num_runs=5

# 循环执行程序
for ((i=1; i<=$num_runs; i++))
do
    # 【nohup(no hang up) 终端关闭不挂起而继续运行(类似于screen)】 
    # + 运行程序 + 传入参数 + 【2>&1 日志重定向】
    # +【& 后台运行实现并发】
    nohup python run.py $i > log_${i}.txt 2>&1 &
done

# 等待所有子线程完成后在继续主线程，否则会在主线程结束时直接结束所有子线程
wait

# 输出提示信息
echo "所有程序执行完成。"
```