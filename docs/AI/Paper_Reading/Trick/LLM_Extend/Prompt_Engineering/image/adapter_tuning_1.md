## Adapter Tuning
> 论文：Parameter-Efficient Transfer Learning for NLP  
> Google Research & Jagiellonian University, 2019 Feb, ICML 2019

### 主要内容
- Adapter modules yield a compact and extensible model; they add only a few trainable parameters per task
- Adapter-based tuning requires training two orders of magnitude fewer parameters to fine-tuning, while attaining similar performance
- Adapters are new modules added between layers of a pre-trained network
- $\phi_{w, v}(x)$，其中$\phi_{w, v_0}(x) \approx \phi_w(x), \vert v \vert \ll \vert w \vert$, only task-specific parameters $v$ and LN layer trained. (training the layer normalization parameters alone is insufficient for good performance)
- Adapter modules have two main features: a small number of parameters, and a near-identity initialization, We also observe that if the initialization deviates too far from the identity function, the model may fail to train.
- Adapter internal skip-connection: with the skip-connection, if the parameters of the projection layers are initialized to near-zero, the module is initialized to an approximate identity function.
- insect to all layer
- 该方法能够在只额外对增加的 3.6% 参数规模（相比原来预训练模型的参数量）的情况下取得和Full-Finetuning 接近的效果（GLUE指标在0.4%以内）
- 3.1. Experimental Settings