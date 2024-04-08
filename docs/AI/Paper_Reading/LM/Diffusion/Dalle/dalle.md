DALLE，名称来源于著名画家达利（**Da**lí）和机器人总动员（Wa**ll-E**），是Open AI推出的一个可以根据书面文字生成图像的人工智能系统。

## DALL-E-1

## DALL-E-2
也称作unCLIP，即CLIP的逆过程：{>>通过给定imge_hidden_state生成图片<<}。层级式地生成64\*64 -> 256\*256 -> 1024\*1024分辨率的图片，逐渐高清
### Framework
<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Diffusion\Dalle\image\dalle-2_framework.png" style="width: 100%;">
    <p style="text-align: center;">unCLIP概略图</p>
</div>

DALLE-2模型包含**prior**和**decoder**两个模块，前者基于输入的文本生成相应的image_hidden_state，后者根据图-文特征生成图片，模型包含的主要技术有：

1. 对比学习（CLIP训练时使用的方法）
2. 多模态（图-文模态信息信息交互、转换）
3. 扩散模型（prior以及decoder模块主要思路）
#### prior
输入描述文本，输出相应的image_hidden_state

1. 给定文本输入，通过（frozen） clip获取图片-文本对$(x, y)$的文本特征$z_i$和$z_t$（中间特征用作dalle-2 prior的ground-truth，即guidence）


扩散模型，一个图片，分批次向里面加噪声，最后真变成了一个噪声，称作前向扩散forward diffusion；该过程的反向模型，通过噪声还原真实图片的过程就是逆向扩散reverse diffusion，直接从噪声恢复图片的难度是相当大的，所以需要类似于chain-of-thought来step by step恢复
 - https://wangjia184.github.io/diffusion_model/
 - single step forward: 扩散模型建模
 - multiple steps forward: 扩散模型正向过程，以及x_t向随机噪声变化的过程
 - 逆向过程，预测正态分布ε以完成基于扩散模型的图像反向构建恢复
 - 所有的步骤采样都基于同一个正态分布ε，然后通过反向过程采样进行拟合
 - loss：x_0与\tilede{x_0}的cross-entropy和其他time_steps的正向x_t与逆向\tilde{x_t}的KL散度，训练的本质上是希望【正向和反向是互通的】
 - loss中  \epsilon_{\theata}(a*b + b\epsilon)指的是q(x_t|x_0, t)估计出来的\miu


- **train**
    1. a
    1. a
    1. s

- **infer**
    1. b
    1. b
    1. b

#### decoder
- **train**
    1. a
    1. a
    1. s

- **infer**
    1. b
    1. b
    1. b

### Result

#### Limitation & Risk

### Discovering
#### Hidden Vocabulary



<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Diffusion\Dalle\image\dalle-2_language_farmer&talk.png" style="width: 100%;">
    <!-- <p style="text-align: center;">unCLIP概略图</p> -->
</div>


<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Diffusion\Dalle\image\dalle-2_language_birds&bugs.png" style="width: 100%;">
    <!-- <p style="text-align: center;">unCLIP概略图</p> -->
</div>

<div class="one-image-container">
    <img src="\AI\Paper_Reading\LM\Diffusion\Dalle\image\dalle-2_language_whale&talk.png" style="width: 100%;">
    <!-- <p style="text-align: center;">unCLIP概略图</p> -->
</div>

!!! danger
    由于细节的自然语言表达性无法保证，DALLE语言存在被“破解”用作非法用途的风险








## DALL-E-3