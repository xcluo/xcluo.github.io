- Sigmoid
$$
\text{Sigmoid}(x)=\sigma(x) = \frac{1}{1+e^{-x}}
$$
- Tanh
$$
\text{Tanh}(x)=\frac{e^x - e^{-x} }{e^x + e^{-x} }=\frac{2}{1+e^{-2x}} -1
$$
- ReLU(Rectified-Linear Units)
$$
\text{ReLU}(x)=\max(0, x)
$$
- Leaky ReLU
$$
\text{Leaky ReLU}(x, \alpha)=\max(0, x) + \alpha\min(0, x)
$$
- GELU(Gaussian Error Linear Units)

$$
\begin{aligned}
\text{GELU}(x)=xP(X\le x)=x\Phi(x) &= 0.5x[1+\text{erf}({x}/{\sqrt{2}})] \\
& \approx x\sigma(1.702x) \\
& \approx 0.5x\Big[ 1 + \text{Tanh}(\sqrt{\frac{2}{\pi}} \big(x+ 0.044715x^3)\big) \Big] 
\end{aligned}
$$

- Swish
$$
\text{Swish}_\beta(x, \beta)=x\sigma(\beta x)
$$
- Mish


https://arxiv.org/pdf/2002.05202
- GLU(Gated Linear Units)
$$
\text{GLU}(x, W, V, b, c)=\sigma(xW+b)\otimes(xV+c)
$$
- Bilinear
$$
\text{Biliniear}(x, W, V, b, c)=(xW+b)\otimes(xV+c)
$$
- ReGLU
$$
\text{ReGLU}(x, W, V, b, c)=\max(0, xW+b)\otimes(xV+c)
$$
- GeGLU
$$
\text{GeGLU}(x, W, V, b, c)=\text{GELU}(xW+b)\otimes(xV+c)
$$
- SwiGLU
$$
\text{SwiGLU}(x, W, V, b, c)=\text{Swish}_\beta(xW+b)\otimes(xV+c)
$$
