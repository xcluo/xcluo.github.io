#### Jacobian Matrix
https://zhuanlan.zhihu.com/p/138334587  
https://zhuanlan.zhihu.com/p/90496291/

雅可比矩阵，$f:\mathbb{R}^n \rightarrow \mathbb{R}^m，J_f(x)\in \mathbb{R}^{m*n}，J_{ij}=\frac{\partial f_i}{\partial x_j}$

$$
J =\big[\frac{\partial f}{\partial x_1} \dots \frac{\partial f}{\partial x_n}\big] = \begin{bmatrix}
    \frac{\partial f_1}{\partial x_1} & \dots & \frac{\partial f_1}{\partial x_n} \\
    \vdots  & \ddots & \vdots  \\
    \frac{\partial f_m}{\partial x_1} & \dots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

#### Hessian matrix
黑塞矩阵，又称作海森矩阵、海塞矩阵或海瑟矩阵，$H_{ij}=I_{ij}=\frac{\partial^2 f}{\partial x_i \partial x_j}$


$$
H = \begin{bmatrix}
    \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1\partial x_2} &\dots & \frac{\partial^2 f}{\partial x_1\partial x_n} \\
    \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} &\dots & \frac{\partial^2 f}{\partial x_2\partial x_n} \\
    \vdots  & \vdots  & \ddots & \vdots  \\
    \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} &\dots & \frac{\partial^2 f}{\partial x_n^2} \\
\end{bmatrix}
$$