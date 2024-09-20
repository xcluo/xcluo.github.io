### 单步激活函数
####  Sigmoid
$$
\text{Sigmoid}(x)=\sigma(x) = \frac{1}{1+e^{-x}}
$$
####  Tanh
$$
\text{Tanh}(x)=\frac{e^x - e^{-x} }{e^x + e^{-x} }=\frac{2}{1+e^{-2x}} -1
$$
####  ReLU
1. ReLU(Rectified-Linear Units)

$$
\text{ReLU}(x)=\max(0, x)
$$

2. Leaky ReLU

$$
\text{Leaky ReLU}(x, \alpha)=\max(0, x) + \alpha\min(0, x)
$$


### 门限激活函数

#### Mish

$$
\text{Mish}(x, \alpha)=x*\text{Tanh}\big(\ln(1+e^x)\big)
$$


####  GELU
(Gaussian Error Linear Units)

$$
\begin{aligned}
\text{GELU}(x)=xP(X\le x)=x\Phi(x) &= 0.5x[1+\text{erf}({x}/{\sqrt{2}})] \\
& \approx x\sigma(1.702x) \\
& \approx 0.5x\Big[ 1 + \text{Tanh}(\sqrt{\frac{2}{\pi}} \big(x+ 0.044715x^3)\big) \Big] 
\end{aligned}
$$


!!! info ""
    sigmoid近似精度低，计算快；tanh近似精度高，计算慢。


#### Swish

$$
\text{Swish}_\beta(x, \beta)=x\sigma(\beta x)
$$


#### GLU
(Gated Linear Units)


$$
\text{GLU}(x, W, V, b, c)=\sigma(xW+b)\otimes(xV+c)
$$

#### Bilinear

$$
\text{Biliniear}(x, W, V, b, c)=(xW+b)\otimes(xV+c)
$$

#### ReGLU

$$
\text{ReGLU}(x, W, V, b, c)=\max(0, xW+b)\otimes(xV+c)
$$

#### GeGLU

$$
\text{GeGLU}(x, W, V, b, c)=\text{GELU}(xW+b)\otimes(xV+c)
$$

#### SwiGLU

$$
\text{SwiGLU}(x, W, V, b, c)=\text{Swish}_\beta(xW+b)\otimes(xV+c)
$$

!!! info ""
    - 门限激活函数进一步细分激活步骤，一部分负责控制门，另一部分则负责生成门的输入值。
    - Google工作[GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202)通过对比实验展示门限激活函数的优越表现。


