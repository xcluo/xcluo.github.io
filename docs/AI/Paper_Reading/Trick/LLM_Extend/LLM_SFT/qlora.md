Quantized Low-Rank Adaptation

1. 4-bit normalfloat, better than b-bit integers and 4-bti floats
2. double quantization, average 0.37-bit per parameter
3. paged optimizers, using NVIDIA unified memory to avoid the gradient checkpointing memory spikes