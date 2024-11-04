### DeBERTa_v1
> 论文：DeBERTa: **D**coding-**e**nhanced **BERT** with disentangled **a**ttention  
> Microsoft Dynamics 365 AI & MSR, ICLR 2021

#### 工作亮点
- 提升BERT和RoBERTa模型效果
- disentangled(解开、解耦，指分解操作【emb_total → pos_emb, token_emb】) attention mechanism
    - each layer, a token is comprised of content and relative position vectors rather than their sum
    - attention weights are computed using disentangled matrices on two vectors
- enhanced mask decoder(EMD)，$P\in\mathbb{R}^{2k*d}$
    - consider absoluate position when decoding the masked words before softmax layer
    - enhanced MLM
- virtual adversarial training (VAT) when finetune to improving models’ generalization.
    - Scale-invariant-Fine-Tuning, a regularization method for improving model's generalization
    - 先进行normalization，再添加扰动噪声
- share relative position embedding projection matrices with $W_q, W_k$
- add convolution layer (induces n-gram knowledge) aside 1-st layer and sum-up together feeding into next layer
- DeBERTa-MT pretrained using MLM and auto-regressive as in UniLM

### DeBERTa_v3
> 论文：DeBERTav3: improving DeBERTa using electra-style pre-training with gradient-disentangled embedding sharing  
> AI & MSR, ICLR 2023


#### 工作亮点
- Embedding Sharing (ES), No Embedding Sharing (NES), Gradient-Disentangled Embedding Sharing (GDES)
- table 2、9, ablation embedding sharing methods