#### PP
- layer-wise model parallelism
- https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/collective/collective_mp/pipeline.html

流水线并行Pipeline Parallel将模型按层分割到不同设备，形成处理流水线。传统PP为单mini-batch时序运行，同一时刻只有一个stage工作，效率低下

- PipeDream
- GPipe

![alt text](image.png)