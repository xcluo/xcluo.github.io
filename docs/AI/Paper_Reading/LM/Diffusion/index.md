


Diffusion 扩散模型



DDPM（Denoising Diffusion Probabilistic Models），单步，即x_t = f(x_{t-1})，预测图像间的噪声，而不是预测时间步的图像本身
DDIM：直接x_t = f(x_0)

spherical interpolation：球面线性插值

、improved DDPM

diffusion beats gan

glide模型、classifier guided diffusion、classifier free guidence


Dalle-2，hierarchical 层级式地生成图片，64*64 -> 256*256 -> 1024*1024，逐渐高清

扩散模型：又逼真又多样，因为是分布模型，基于采样呈现多样性
GAN：逼真（优化目标为以假乱真）但不多样

#### train
1. 给定文本输入，通过（frozen） clip获取图片-文本对$(x, y)$的文本特征$z_i$和$z_t$（中间特征用作dalle-2 prior的ground-truth，即guidence）



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


unCLIP

扩散模型，一个图片，分批次向里面加噪声，最后真变成了一个噪声，称作前向扩散forward diffusion；该过程的反向模型，通过噪声还原真实图片的过程就是逆向扩散reverse diffusion，直接从噪声恢复图片的难度是相当大的，所以需要类似于chain-of-thought来step by step恢复
 - https://wangjia184.github.io/diffusion_model/
 - single step forward: 扩散模型建模
 - multiple steps forward: 扩散模型正向过程，以及x_t向随机噪声变化的过程
 - 逆向过程，预测正态分布ε以完成基于扩散模型的图像反向构建恢复
 - 所有的步骤采样都基于同一个正态分布ε，然后通过反向过程采样进行拟合
 - loss：x_0与\tilede{x_0}的cross-entropy和其他time_steps的正向x_t与逆向\tilde{x_t}的KL散度，训练的本质上是希望【正向和反向是互通的】


beta的schedule，线性、cosine等

每一步加入噪声的过程互相独立，因此正态分布状态可以叠加

多层采样逆推还原


dalle 语言

UNet
   - 右下角和图例：卷积、上下采样、skip connect concat
   - down sample，结果图层通道变多、图变小：2*2最大池化，为保留更多特征，也可以用3*3卷积
   - up sample，结果图层通道变小、图变多：转置卷积（类似于填充了空洞卷积的形式让图片变大），也可以插值法（比如填充邻近像素或周围像素均值等）

加载pytorch checkpoint，打印模型结构

web paint online

pad=reflect，填充的像素以边界为对称轴对称，由于是非零，因此pad后整张图所有像素都有特征