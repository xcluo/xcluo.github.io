### FGSM
> 论文：Explaining and Harnessing Adversarial Examples  
> FGSM：**F**ast **G**radient **S**ign **M**ethod  
> Google, ICLA 2015

#### 工作要点
- fgsm attack，在梯度方向些微扰动（r扰动perturbation.或噪声noise，或者说是残差residual）即可轻易使模型严重误判
- 【梯度上升方向（+noise）降低置信、梯度下降方向（-noise）提升置信】，属于白盒攻击，还可以对梯度进一步应用l2 norm，即$\epsilon\frac{g}{||g||_2}$，https://jaketae.github.io/study/fgsm/
- 通过对【对抗样本 + 干净样本】的数据混合训练，神经网络能实现一定程度上的正则化与泛化增强，即下式
- $\hat{J}(\theta, x, y)=\alpha J(\theta, x, y) + (1-\alpha)J(\theta, x+\eta*sign(\nabla_x J(\theta, x, y)), y)$，$\alpha$ 一般取0.5
- 从上式可以发现，对于每个训练数据附近的一个临域内，我们都可以保持它的识别正确。这样模型的鲁棒性也有了一定的提升。由于没有显示地制造数据参与训练，而是每次对输入representation（可以为embedding也可以为其它中间状态）进行动态虚拟制造，因此叫做VAT
- 直接沿用标签

### LDS
> 论文：Distributional Smoothing with Virtual Adversarial Training  
> LDS：**L**ocal **D**istributional **S**moothness  
> Kyoto University, ICLA 2016

#### 工作要点
- 最多两个超参：$\epsilon \gt 0\ \&\ ||r||_2 \lt \epsilon$、$\lambda$用于连接多个loss object
- 3次计算，forward_1: f(x), backward_1: d, forward_2: f(x+εd)
- approximation of LDS
- 局部分布平滑性，即$f(x)\approx f(x+r)$
- [LDS推导](https://blog.csdn.net/kearney1995/article/details/79970934)
- https://blog.csdn.net/qwq_xcyyy/article/details/119420855
- [二阶泰勒展开](https://www.cnblogs.com/aoru45/p/13876279.html)
- $r_{v-adv}(x, \theta)\approx \epsilon\overline{u(x,\theta)}$，其中$u(x,\theta)$为 $H(x,\theta)$ 的第一个特征向量（主特征向量，对应于最大特征值），$\overline{\cdotp}$表示单位矩阵化操作
- 算法实际上是$d=\overline{\frac{g}{||g||_2}}$ https://blog.csdn.net/u013453936/article/details/81612015
- 不直接沿用标签，而是对局部分布进行平滑处理（KL min loss）
- [power iteration method](https://blog.csdn.net/qq_44154915/article/details/133957332)

### FGM
> 论文：Adversarial Training Methods for Semi-supervised Text Classification  
> FGM：**F**ast **G**radient **M**ethod  
> Github：[adversarial_text](https://github.com/tensorflow/models/tree/master/research/adversarial_text)  
> Preferred Networks & Google & OpenAI, ICLA 2017

#### 工作要点
- 第一个将VAT应用至文本领域
- dropout followed by FGM performs the best
- 相较fgsm进一步用了l2 norm，$\epsilon\frac{g}{||g||_2}$

### PGD
> 论文：Towards Deep Learning Models Resistant to Adversarial Attacks  
> PGD：**P**rojected **G**radient **D**escent  
> MIT, ICLA 2018

#### 工作要点
- s

### s
> 论文：Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning  
> Preferred Networks & Kyoto University & ATR Cognitive Mechanisms Laboratories & Ritsumeikan University, TPAMI 2019

#### 工作要点
- https://blog.csdn.net/weixin_43301333/article/details/108349415


### SMART
> 论文：SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization  
> SMART：**SM**oothness-inducing **A**dversarial **R**egularization  
> Microsoft Dynamics 365 AI, ACL 2020