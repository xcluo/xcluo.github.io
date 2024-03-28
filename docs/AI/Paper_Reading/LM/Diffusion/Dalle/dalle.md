DALLE，名称来源于著名画家达利（**Da**lí）和机器人总动员（Wa**ll-E**），是Open AI推出的一个可以根据书面文字生成图像的人工智能系统。

### DALL-E-1

### DALL-E-2

每一步加入噪声的过程互相独立，因此正态分布状态可以叠加


Dalle-2也叫unCLIP（从文本或图片还原出图片数据），hierarchical 层级式地生成图片，64*64 -> 256*256 -> 1024*1024，逐渐高清


#### train
1. 给定文本输入，通过（frozen） clip获取图片-文本对$(x, y)$的文本特征$z_i$和$z_t$（中间特征用作dalle-2 prior的ground-truth，即guidence）


扩散模型，一个图片，分批次向里面加噪声，最后真变成了一个噪声，称作前向扩散forward diffusion；该过程的反向模型，通过噪声还原真实图片的过程就是逆向扩散reverse diffusion，直接从噪声恢复图片的难度是相当大的，所以需要类似于chain-of-thought来step by step恢复
 - https://wangjia184.github.io/diffusion_model/
 - single step forward: 扩散模型建模
 - multiple steps forward: 扩散模型正向过程，以及x_t向随机噪声变化的过程
 - 逆向过程，预测正态分布ε以完成基于扩散模型的图像反向构建恢复
 - 所有的步骤采样都基于同一个正态分布ε，然后通过反向过程采样进行拟合
 - loss：x_0与\tilede{x_0}的cross-entropy和其他time_steps的正向x_t与逆向\tilde{x_t}的KL散度，训练的本质上是希望【正向和反向是互通的】
 - loss中  \epsilon_{\theata}(a*b + b\epsilon)指的是q(x_t|x_0, t)估计出来的\miu


dalle 语言


### DALL-E-3