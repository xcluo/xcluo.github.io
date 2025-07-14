- [x] kv cache：https://juejin.cn/post/7362789570217885759
- [x] KV cache：将L层K与V进行缓存以执行Attention，各层矩阵为 `k/v.shape = (bs, head_num, kv_len, head_dim)`
- [x] https://ai.stackexchange.com/questions/48185/why-not-cache-the-q-query-matrix
    - infer时只需要得到`current_last_hidden_state`，不需要关心`previous_last_hidden_state`，因此无需进行 q cache（FFN和Attention均不存在current $t$ 与previous $\lt t$的交互）  
    - Attention部分存在$QK^TV$，因此需要持续缓存K、V
    - during autoregressive generation you do not use the entire cached input X
 to recompute Q ; instead, you compute the projection for the new token $x_\text{new}$ on the fly to get $q_\text{new}$, $q_\text{new}K^TV$
    - 在AR模型的infer阶段，只需要输入$x_t$与其对应的位置信息，结合缓存的K、V cache，即可实现next token prediction

- kv cache 量化