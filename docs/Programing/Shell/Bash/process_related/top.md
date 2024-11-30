[top命令结果解读](https://blog.csdn.net/fracly/article/details/129789900)

#### line_3
`%Cpu(s): 9.6 us, 0.8 sy, 0.0 ni, 89.6 id, 0.0 wa, 0.0 hi, 0.0 si, 0.0 st`，表示CPU使用情况，均为百分比

- `9.6 us` 用户空间（user）占用CPU百分比
- `0.8 sy` 内核空间（system）占用CPU百分比
- `0.0%ni` 用户进程空间内改变过优先级的进程占用CPU百分比
- `89.6 id` 空闲CPU百分比
- `0.0%wa` 等待（wait）输入输出的CPU时间百分比
- `0.0%hi` 硬中断（Hardware IRQ）占用CPU的百分比
- `0.0%si` 软中断（Software Interrupts）占用CPU的百分比
- `0.0 st` 用于有虚拟cpu的情况，用来指示被虚拟机偷掉的cpu时间
#### line_4-5
```bash
KiB Mem : 65807304 total, 432360 free, 37334904 used, 28040040 buff/cache
KiB Swap: 0 total, 0 free, 0 used. 26442184 avail Mem
```
两行均为为内存统计情况，
- `KiB` 单位，有时也会显示 `MiB`
- `Mem` 为实际物理内存使用情况，`Swap`是虚拟内存swap的情况
- `total` 总内存空间
- `free` 空间内存空间
- `used` 已使用内存空间  
- `buff/cache` 用作内存缓存的内存量

#### line_6
`PID USER PR NI VIRT RES SHR S %CPU %MEM TIME+ COMMAND`，每列参数意义

- `USER` 进程拥有者
- `PR` 进程优先级PR=20+NI，默认为20
- `NI` nice好人值，值越大越容易被插队，默认0
- `VIRT` 进程使用虚拟内存总量，VIRT=SWAP+RES
- `SHR` 共享内存大小
- `S` 进程状态，{D: 不可中断的睡眠状态, R: 运行, S: 睡眠, T: 跟踪/停止, Z: 僵尸进程, I: 空间进程}
- `%CPU` 上次更新到现在的CPU时间占用百分比
- `%MEM` 进程使用的物理内存百分比
