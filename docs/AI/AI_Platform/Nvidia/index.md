- cuda
## Nvidia GPU 计算架构
1. Tesla
2. Fermi
3. Kepler
4. Maxwell
5. Pascal
6. Volta
7. Turing
8. Ampere
9. Ada Lovelace
10. Hopper
11. Blackwell



- triton, CUTLASS(CUDA Templates for Linear Algebra Subroutines)
- CUTLASS profiler tool
- Warp Matrix Multiply and Accumulate (WMMA) 是现代 NVIDIA GPU（特别是从 Volta 架构开始）上实现极致计算性能的核心技术，通常指WMMA api，允许开发者以 Warp（32个并行线程的组）为执行单位，直接对小型矩阵（例如 16x16, 32x8 等，称为Tile或Fragment）进行高效的乘加运算。
- thread-blocks, WARPs, and WMMA (Tensor cores)
- cuDNN
- Megatron-AI威震天

- 算力单位
- **TFLOPs**: 1e12次浮点 ^^运算^^ 次数
- **TOPs**：1e12整数运算次数
- Kernel Implementation
