# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (2023)
Source: 90c0ba-2023.pdf

## Core reasons
- FlashAttention-2 reorders attention computation to minimize non-matmul FLOPs and parallelizes the work across thread blocks and warps so that attention stays close to GEMM efficiency, yielding about 2× speedups and up to 72% model FLOPs utilization during GPT-style training (no positional encoding or dimensional changes).
- The paper details how the algorithm and work partitioning exploit block-wise softmax, warp-level cooperation, and thread-block-level parallelism to consistently attain 2-3× speedups on modern GPUs, marking a systems-level optimization of the Transformer core compute.

## Evidence extracts
- "We propose FlashAttention-2, with better work partitioning to address these issues. In particular, we (1) tweak the algorithm to reduce the number of non-matmul FLOPs (2) parallelize the attention computation, even for a single head, across different thread blocks to increase occupancy, and (3) within each thread block, distribute the work between warps to reduce communication through shared memory. These yield around 2× speedup compared to FlashAttention, reaching 50-73% of the theoretical maximum FLOPs/s on A100 and getting close to the efficiency of GEMM operations." (p. 1)
- "We describe the FlashAttention-2 algorithm, which includes several tweaks to FlashAttention to reduce the number of non-matmul FLOPs. We then describe how to parallelize the computation on different thread blocks to make full use the GPU resources. Finally we describe we partition the work between different warps within one thread block to reduce the amount of shared memory access. These improvements lead to 2-3× speedup as validated in Section 4." (p. 5)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
