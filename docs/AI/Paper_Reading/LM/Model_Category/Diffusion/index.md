Diffusion 扩散模型

扩散模型：又逼真又多样，因为是分布模型，基于采样呈现多样性

   - 隐变量
   
GAN：逼真（优化目标为以假乱真）但不多样

### 扩散模型
#### DDPM
1. [DDPM](./DDPM/DDPM)

#### DALL-E

1. [DALL-E-1](./Dalle/dalle#dalle-e-1)
2. [DALL-E-2](./Dalle/dalle#dalle-e-2)
3. [DALL-E-3](./Dalle/dalle#dalle-e-3)



### Noise Scheduler
1. cosine
2. linear
3. quadratic
4. jsd
5. sigmoid




DDPM（Denoising Diffusion Probabilistic Models），单步，即x_t = f(x_{t-1})，时间步t-1时刻的图像分布，采样得到x_{t-1}

DDIM：直接ε_{t-1} = f(x_0, x_t)

spherical interpolation：球面线性插值

improved DDPM

diffusion beats gan

glide模型、classifier guided diffusion、classifier free guidence






（预测时只需要）Dalle-2两阶段（测试发现两阶段比直接生成效果好很多）：
1. prior模型，通过 $y$ 的 $z_t$ 来学习预测对应的 $z_i$
   - text_eocoding + text_representation + noise_image_representation + learn_query(类似于[CLS]) -- predict --> image_representation
2. decoder，根据 $z_i$（可选择是否使用文本信息 $y$ or $z_t$），级联式地还原生成高清图片 $x$
   - embedding 分割，将1*d分割为h*d/h（相当于一个token细化为多个token）参与Transformer，更好地对每个维度细化权重



AE：mask x后输入至模型，再根据中间特征进行x的还原

DAE：较AE在masked_x上加了噪声

V(ariational)AE：不再是学习特征，而是学习预测概率的均值和方差，得到均值和方差采样一个中间特征，以实现x的还原

V(ector)Q(uantised)VAE：VAE量化，将连续分布离散化（比如一个codebook $\in K\times D$，只需要使用目标向量（比如最相似）作为输入

pixel cnn: 用cnn来学习特征,然后用VAE来生成图片


beta的schedule，线性、cosine等


UNet
   - 右下角和图例：卷积、上下采样、skip connect concat
   - down sample，结果图层通道变多、图变小：2*2最大池化，为保留更多特征，也可以用3*3卷积
   - up sample，结果图层通道变小、图变多：转置卷积（类似于填充了空洞卷积的形式让图片变大），也可以插值法（比如填充邻近像素或周围像素均值等）


pad=reflect，填充的像素以边界为对称轴对称，由于是非零，因此pad后整张图所有像素都有特征