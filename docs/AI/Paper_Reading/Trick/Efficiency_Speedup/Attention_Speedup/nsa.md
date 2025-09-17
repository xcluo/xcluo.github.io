## NSA
> 论文：Native Sparse Attention: Hardware-Aligned and **N**atively Trainable **S**parse **A**ttention  
> DeepSeek-AI & Peking University & University of Washington, 2025 Feb, ACL 2025

### 主要内容
- 本质上为hierarchical attention
- NSA employs a dynamic hierarchical sparse strategy, combining coarse-grained token compression with fine-grained token selection to preserve both global context awareness and local precision
- attention computation with softmax architectures accounts for 70-80% of total latency when decoding 64k-length contexts, underscoring the urgent need for more efficient attention mechanisms
- Hardware-aligned inference speedup；Training-aware algorithm design
- To achieve more effective and efficient sparse attention