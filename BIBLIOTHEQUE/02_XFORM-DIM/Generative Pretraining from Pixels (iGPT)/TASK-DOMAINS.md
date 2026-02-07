# Generative Pretraining from Pixels (Not specified in the paper)
Source: Generative Pretraining from Pixels (iGPT).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generation (autoregressive pixel prediction) | images (pixels) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | pixels (next-pixel logits) | 2D (x, y) (inferred) | Fixed (inferred) |
| generation (masked pixel prediction) | images with masked pixels | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | masked pixel values (logits) | 2D (x, y) (inferred) | Fixed (inferred) |
| classification (image classification) | images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | class logits / labels | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers image-domain tasks including autoregressive pixel prediction, masked pixel prediction, and downstream image classification via fine-tuning or linear probes. Inputs are images resized to fixed low resolutions and reshaped into sequences, so the tasks operate over fixed-size 2D image grids and fixed output sizes (inferred). The transformer uses fixed-context self-attention and a direct mapping from input sequences to outputs without external state described (static/direct, inferred).

## Evidence
### Task: generation (autoregressive pixel prediction)
- "We train a sequence Transformer to auto-regressively predict pixels, without incorporating knowledge of the 2D input structure." (Abstract)
- "First, we pre-process raw images by resizing to a low resolution and reshaping into a 1D sequence." (Figure 1 caption)
- "learn a projection from  $n^L$  to logits parameterizing the conditional distributions at each sequence element." (Section 2.2)
- "Our models have IRs of either  $32^2 \times 3$ ,  $48^2 \times 3$ , or  $64^2 \times 3$ ." (Section 3.2)
- "we apply the standard upper triangular mask to the  $n \times n$  matrix of attention logits." (Section 2.2)
- Inference: In/Out Dimension labeled 2D (x, y) and In/Out Dynamics labeled Fixed because the task operates on images with a referenced "2D input structure" and fixed IRs; Attention labeled Static because the model applies a fixed attention mask; State labeled Direct because it maps an input sequence directly to per-position logits with no external state described (Abstract; Sections 2.2, 3.2).

### Task: generation (masked pixel prediction)
- "We then chose one of two pre-training objectives, auto-regressive next pixel prediction or masked pixel prediction." (Figure 1 caption)
- "We also consider the BERT objective, which samples a sub-sequence  $M \subset [1,n]$" (Section 2.1)
- "minimizing the negative log-likelihood of the \"masked\" elements  $x_M$  conditioned on the \"unmasked\" ones" (Section 2.1)
- "When training BERT, we simply ignore the logits at unmasked positions." (Section 2.2)
- "When using the BERT objective, no attention logit masking is required." (Section 2.2)
- "Our models have IRs of either  $32^2 \times 3$ ,  $48^2 \times 3$ , or  $64^2 \times 3$ ." (Section 3.2)
- Inference: In/Out Dimension labeled 2D (x, y) and In/Out Dynamics labeled Fixed because the task operates on images with fixed IRs; Attention labeled Static because the model uses fixed attention masking; State labeled Direct because the model maps the masked input sequence directly to logits without external state described (Abstract; Sections 2.2, 3.2).

### Task: classification (image classification)
- "One way to measure representation quality is to fine-tune for image classification." (Section 2)
- "We learn a projection from  $f^L$  to class logits, which we use to minimize a cross entropy loss  $L_{CLF}$ ." (Section 2.3)
- "Then, a linear classifier is trained on  $(f_X, Y)$ ." (Section 2)
- "we average pool  $n^L$  across the sequence dimension to extract a d-dimensional vector of features per example:" (Section 2.3)
- "Our models have IRs of either  $32^2 \times 3$ ,  $48^2 \times 3$ , or  $64^2 \times 3$ ." (Section 3.2)
- Inference: In Dimension labeled 2D (x, y) and In Dynamics labeled Fixed because inputs are images resized to fixed IRs; Attention labeled Static because the model uses fixed attention masking; State labeled Direct because the model maps the input sequence to pooled features and class logits without external state described; Out Dimension labeled 0D because outputs are class logits/labels (Sections 2.2, 2.3, 3.2).
