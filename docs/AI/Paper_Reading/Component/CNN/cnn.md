### 卷积

Convolution

#### 经典卷积

#### 空洞卷积

Dilated Convolution，也叫做膨胀卷积，膨胀率（Dilation Rate）

### 上采样 Upsampling

#### 转置卷积

转置卷积（Transposed Convolution），也叫做反卷积/逆卷积（Deconvolution）、上卷积（Up-Convolution）或分数步长卷积（Fractionally-Strided Convolution），通过逆向操作模拟卷积的逆过程（并非数学上的严格逆运算）实现尺寸放大。大致核心流程如下：

1. 以指定步长（stride）在特征图元素间填充（padding）空值
2. 在该被“撑大”的特征图时上执行卷积操作，得到最终放大后的输出

#### 子像素卷积

- 子像素卷积Sub-Pixel Convolution

### 池化 Pooling

#### 经典池化

- mean, avg, max

#### RoI

1. 在原始图片圈定候选区域，即Region Proposal（如矩形4点）
2. 讲原始图片通过特征映射转换（如卷积，模型等）为特征图
3. 根据图片和特征图大小缩放比例，讲区域缩放到特征图尺寸（若存在碎片，可以通过量化方式解决）
4. 对缩放后的区域进行池化，得到RoI Pooling结果

### 反池化 Unpooling

- Max Unpooling, Average Unpooling