# Stabilizing Equilibrium Models by Jacobian Regularization (2021)
Source: Stabilizing Equilibrium Models by Jacobian Regularization.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scalar function regression | Scalar value `x` | 0D (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Scalar prediction `y` | 0D (inferred) | Fixed (inferred) |
| Word-level language modeling (next-word prediction) | Word/token sequence `x_{1:T}` | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Next-word/token sequence `y_{1:T}` | 1D (t) (inferred) | Capped (inferred) |
| Image classification | Images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Class label | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates DEQ models on three task intents: scalar function regression, word-level language modeling, and image classification. Across these tasks, the justified task-domain coverage spans 0D scalar mappings, 1D (t) token-sequence modeling, and 2D (x, y) image inputs with 0D class outputs. The OCR supports Fixed dynamics for synthetic/image settings and Capped dynamics for language modeling due to explicit sequence-length limits. Attention and state behavior are not explicitly labeled in the paper, and are therefore inferred as Static attention and Direct state for the reported setups.

## Evidence
### Task: Scalar function regression
- "We generated 5096 scalar data pairs (x,y) using function  $y=h(x)=\frac{3}{2}x^3+x^2-5x+2\sin(x)-3+\delta$  (where  $\delta\in\mathcal{N}(0,0.05)$ )." (Section 5.1)
- "To visualize the effect of the proposed Jacobian regularization on DEQ models (see Section 5), we generated a synthetic dataset with 5096 pairs (x, y) from the target function." (Section A.1)
- Inference: `0D`, `Fixed`, `Static`, and `Direct` are inferred from the scalar `x -> y` setup and the absence of any runtime input-selection mechanism in the task description (Sections 5.1 and A.1).

### Task: Word-level language modeling (next-word prediction)
- "Word-level language modeling tasks aim to predict the next word of a textual sequence by integrating the semantics and information of current and past tokens." (Section A.2)
- "Formally, given an input sequence  $\mathbf{x}_{1:T} \in \mathbb{R}^{T \times p}$  (where  $x_i \in \mathbb{R}^p$  and T is the sequence length), an autoregressive sequence model G produces output  $G(\mathbf{x}_{1:T}) = \mathbf{y}_{1:T} \in \mathbb{R}^{T \times q}$." (Section A.2)
- "| Input Sequence Length              | N/A                | 150                            | N/A                        | N/A                       |" (Table 4, Section A)
- Inference: `1D (t)` and `Capped` are inferred from sequence indexing (`x_{1:T}`, `y_{1:T}`) and the explicit sequence-length limit (`150`); `Static` attention and `Direct` state are inferred because no dynamic retrieval or persistent constructed task state is specified (Section A.2 and Table 4).

### Task: Image classification
- "We additionally conduct experiments on vision tasks using the recent multiscale deep equilibrium networks (MDEQ) (Bai et al., 2020)." (Section 5.3)
- "The results of applying Jacobian regularization on multiscale DEQs for image classification are shown in Table 2." (Section 5.3)
- "The CIFAR-10 (Krizhevsky & Hinton, 2009) dataset contains 60,000 color images of resolution  $32 \times 32$  that fall into 10 object classes." (Section A.3)
- "The ImageNet (Krizhevsky et al., 2012) dataset, on the other hand, contains over 1.28M training images and 150K test images, distributed over 1,000 classes. All images are rescaled to  $224 \times 244$  resolution before they are fed into the models." (Section A.3)
- Inference: `2D (x, y)`, `Fixed`, `0D`, `Static`, and `Direct` are inferred from fixed-resolution image inputs and single class decisions, with no described runtime observation-selection or persistent constructed state mechanism (Sections 5.3 and A.3).
