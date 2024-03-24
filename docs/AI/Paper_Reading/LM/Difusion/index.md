
Difusion 扩散模型



DDPM、improved DDPM

difusion beats gan

glide模型、classifier guided difusion、classifier free guidence


Dalle-2，hierarchical 层级式地生成图片，64*64 -> 128*128 -> 256*256 -> 1024*1024，逐渐高清

扩散模型：又逼真又多样，因为是分布模型，基于采样呈现多样性
GAN：逼真（优化目标为以假乱真）但不多样

#### train
1. 给定文本输入，通过（frozen） clip获取图片-文本对$(x, y)$的文本特征$z_i$和$z_t$（中间特征用作dalle-2 prior的ground-truth，即guidence）



（预测时只需要）Dalle-2两阶段（测试发现两阶段比直接生成效果好很多）：
1. prior模型，通过 $y$ 的 $z_t$ 来学习预测对应的 $z_i$
2. decoder，根据 $z_i$（可选择是否使用文本信息 $y$ or $z_t$），级联式地还原生成高清图片 $x$
 


AE：mask x后输入至模型，再根据中间特征进行x的还原

DAE：较AE在masked_x上加了噪声

V(ariational)AE：不再是学习特征，而是学习预测概率的均值和方差，得到均值和方差采样一个中间特征，以实现x的还原

V(ector)Q(uantised)VAE：VAE量化，将连续分布离散化（比如一个codebook $\in K\times D$，只需要使用目标向量（比如最相似）作为输入

pixel cnn: 用cnn来学习特征,然后用VAE来生成图片


unCLIP

扩散模型，一个图片，分批次向里面加噪声，最后真变成了一个噪声，称作前向扩散forward difusion；该过程的反向模型，通过噪声还原真实图片的过程就是逆向扩散reverse difusion，类似于chain-of-thought step by step

多层采样逆推还原