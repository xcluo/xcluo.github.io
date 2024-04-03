### BN
即 Batch Normalization

### LN
即 Layer Normalization
### RMSNorm
即 Root Mean Squared Layer Normalization，==RMS认为LN取得的成功是缩放不变性，而不是平移不变性，因此较LN只保留了缩放（除以标准差）==，去除了计算过程中的平移（分子减去均值）

$$\text{RMS}(x_i)=\frac{x_i}{\sqrt{\frac{1}{d}\sum_{1}^{d}x_i^2}+\epsilon}$$