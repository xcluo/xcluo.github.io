### FGSM
> 论文：Explaining and Harnessing Adversarial Examples  
> FGSM：**F**ast **G**radient **S**ign **M**ethod  
> Google, ICLA 2015

#### 工作要点
- fgsm attack，在梯度方向些微扰动（r扰动perturbation.或噪声noise，或者说是残差residual）即可轻易使模型严重误判
- 【梯度上升方向（+noise）降低置信、梯度下降方向（-noise）提升置信】，属于白盒攻击，还可以对梯度进一步应用l2 norm，即$\epsilon\frac{g}{||g||_2}$，https://jaketae.github.io/study/fgsm/
- 可以理解为l_∞约束，$\frac{g}{||g||_\infty}$，只不过权值不是归一化而是进行sign变换
- 通过对【对抗样本 + 干净样本】的数据混合训练，神经网络能实现一定程度上的正则化与泛化增强，即下式
- $\hat{J}(\theta, x, y)=\alpha J(\theta, x, y) + (1-\alpha)J(\theta, x+\eta*sign(\nabla_x J(\theta, x, y)), y)$，$\alpha$ 一般取0.5
- 从上式可以发现，对于每个训练数据附近的一个临域内，我们都可以保持它的识别正确。这样模型的鲁棒性也有了一定的提升。由于没有显示地制造数据参与训练，而是每次对输入representation（可以为embedding也可以为其它中间状态）进行动态虚拟制造，因此叫做VAT
- 直接沿用标签

### approximation of LDS
> 论文：Distributional Smoothing with Virtual Adversarial Training  
> LDS：**L**ocal **D**istributional **S**moothness  
> Kyoto University, ICLA 2016

#### 工作要点
- 最多两个超参：$\epsilon \gt 0\ \&\ ||r||_2 \lt \epsilon$、$\lambda$用于连接多个loss object
- 3次计算，forward_1: f(x), backward_1: d, forward_2: f(x+εd)
- approximation of LDS
- invariant
- 局部分布平滑性，即$f(x)\approx f(x+r)$
- [LDS推导](https://blog.csdn.net/kearney1995/article/details/79970934)
- https://blog.csdn.net/qwq_xcyyy/article/details/119420855
- [二阶泰勒展开](https://www.cnblogs.com/aoru45/p/13876279.html)
- $r_{v-adv}(x, \theta)\approx \epsilon\overline{u(x,\theta)}$，其中$u(x,\theta)$为 $H(x,\theta)$ 的第一个特征向量（主特征向量，对应于最大特征值），$\overline{\cdotp}$表示单位矩阵化操作
- 算法实际上是$d=\overline{\frac{g}{||g||_2}}$ https://blog.csdn.net/u013453936/article/details/81612015
- 不直接沿用标签，而是对局部分布进行平滑处理（KL min loss）
- [power iteration method](https://blog.csdn.net/qq_44154915/article/details/133957332)

> 论文：Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning  
> Preferred Networks & Kyoto University & ATR Cognitive Mechanisms Laboratories & Ritsumeikan University, TPAMI 2017

#### 工作要点
- https://blog.csdn.net/weixin_43301333/article/details/108349415
- 更细节的实验对比
- 应用了N_unlabel = N - N_label - N_test参与KL散度的部分的smooth阶段，进行半监督学习，搭配figure 1理解
- k值选取对效果影响不大，一般取1就足够
- α和ε选取，前者一般固定为1，后者需要作为超参调整
- isotropically，随机各方向散射

### FGM
> 论文：Adversarial Training Methods for Semi-supervised Text Classification  
> FGM：**F**ast **G**radient **M**ethod  
> Github：[adversarial_text](https://github.com/tensorflow/models/tree/master/research/adversarial_text)  
> Preferred Networks & Google & OpenAI, ICLA 2017

#### 工作要点
- 第一个将VAT应用至文本领域
- dropout followed by FGM performs the best
- 相较fgsm和lds进一步用了l2 norm，$\epsilon\frac{g}{||g||_2}$
- at + vat一起应用，可以共同提升模型的robustness
- table 3，应用at或vat后word embedding的差异性和相似性得到提升

### TextBugger
> 论文：TextBugger: Generating Adversarial Text Against Real-world Applications  
> Zhejiang University & Alibaba-Zhejiang University Joint Research, NDSS 2019

#### 工作要点
- general attack framework for generating adversarial texts
- 白盒环境：
    - 基于雅可比矩阵，并按照梯度降序argidx排序token序列  
    - 遍历有序token序列，当扰动(char-level或word-level)后$S(x, x^{'})\gt \epsilon\ \text{and} \ F(x^{'})\ne y$，返回$x^{'}$
        - insert、delete、swap、subsitute-char and subsitute-word
        - algorithm 2
- 黑盒环境：选择重要性最高的句子，利用打分函数定位并操纵重要tokens
        - algorithm 3

### TextAttack
TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP

### PGD
> 论文：Towards Deep Learning Models Resistant to Adversarial Attacks  
> PGD：**P**rojected **G**radient **D**escent  
> Github：[mnist_challenge](https://github.com/MadryLab/mnist_challenge)、[cifar10_challenge](https://github.com/MadryLab/cifar10_challenge)  
> MIT, ICLA 2018

#### 工作要点
- saddle point (min-max) formulation，通过“小步走，走多步”的鞍点min-max优化方式（FGSM和FGM的优化路径是线性），非线性模型只做一次下降是不够的
- $r_{adv|t+1}=\alpha \frac{g_t}{||g_t||_2}$
- epsilon: maximum perturbation, alpha: step size, steps: number of steps
- start: 是否从随机点开始扰动 `adv_img = img + torch.empty_like(x).uniform_(-eps, eps)`
- 每次更新需要用l∞约束扰动规模，即 `delta=torch.clamp(delta, min=-eps, max=eps), adv_img = img+delta`
- l∞约束比l2约束效果更好，因为后者效果太强了，脱离了对抗的范围，即对抗生成的标签较原真实标签发生了改变  
- https://github.com/Harry24k/adversarial-attacks-pytorch/blob/cf21e81f3f2e7a859e029e3d7953290ef063d6dd/torchattacks/attacks/pgd.py#L74  
- [PGD L2/infinity](https://blog.csdn.net/Sankkl1/article/details/134215790)  
- 不直接输入x_embedding，把fgsm在同一batch内计算n次后（连乘更新），进行min loss，从而解决non-convex（非凸）和non-concave（非凹）点优化问题
- pgd attack，运用局部一阶信息进行强力对抗攻击
- MNIST和CIFAR10白盒攻击效果测试分别为89%/46%，黑盒效果为95%/64%
- Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse 
Parameter-free Attacks
- l∞ ball around 范式约束


### FreeAT
Adversarial Training for Free!
- https://zhuanlan.zhihu.com/p/103593948

### YOPO
You Only Propagate Once


### FreeLB
Free Large-Batch

### SMART
> 论文：SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization  
> SMAR$^3$T$^2$：**SM**oothness-inducing **A**dversarial **R**egularization and b**R**egman p**R**oximal poin**T** op**T**imization  
> Microsoft Dynamics 365 AI, ACL 2020


#### 工作要点
1. smoothness-inducing regularization
        - $\lambda_s$LDS + $\mu$PGD
2. bregman proximal point optimization method (including vinilla, generalized, accelerated proximal and other variants)introduce a trust-region-type regularization at each iteration
        - reltaed to FreeLB  
        - vanilla Bregman proximal point(VBPP)，$\theta_{t+1}=\argmin_{\theta} \mathcal{F}(\theta) + \mu\mathcal{D}_{\text{Breg}}(\theta, \theta_t)$    
        - $\mathcal{D}_{\text{Beeg}}(\theta, \theta_t)=\frac{1}{n}\sum_i^n\mathcal{l}_s(f(x_i;\theta), f(x_i; \theta_t))$  
        - $\mathcal{l}_s$，分类：pgd + LDS，回归：pgd + MSE  
        - momentum Bregman proximal point (MBPP)