## TensorRT

TensorRT 是 NVIDIA 推出的高性能深度学习推理（Inference）优化器和运行时库，专门用于在生产环境中高效部署深度学习模型。该库核心目标是通过一系列优化技术，与Nvidia GPU结合，最大限度地提高基于 Tensorflow、Caffe、Mxnet和Pytorch 等深度学习框架的模型推理性能。



### 安装
https://zhuanlan.zhihu.com/p/706873079
### 主要特性

#### Layer Fusion
层间融合：合并多个操作较少内核调用

#### Precision Calibration
数据精度校准：对量化模型进一步校准

#### Kernel Auto-Tuning
内核自动调优：选择最优内核实现

#### Dynamic Tensor Memory
动态张量内存：最小化内存占用

#### Graph Optimization
图优化：消除冗余计算