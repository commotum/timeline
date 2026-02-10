# The Quantization Model of Neural Scaling (Not specified in the paper)
Source: The Quantization Model of Neural Scaling.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next-token prediction (language modeling) | token contexts from text corpora | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | next token | 0D (inferred) | Fixed |
| binary classification (multitask sparse parity) | bit strings with control bits and task bits | 1D (t) (inferred) | Fixed | Static (inferred) | Direct (inferred) | parity bit (binary label) | 0D | Fixed |
| clustering (quanta discovery from gradients) | next-token samples (token + context) and per-sample gradients | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | cluster labels over samples (quanta clusters) | 0D (inferred) | Capped (inferred) |

## Summary
The paper covers two prediction tasks and one unsupervised analysis task: next-token language modeling, multitask sparse-parity classification, and gradient-based clustering for quanta discovery. Inputs are primarily 1D sequences (tokens or bit strings), while outputs are mostly 0D single-token/label decisions, with clustering producing discrete sample labels. Dynamics are Fixed for sparse parity and output labels, and Capped where transformer context limits or fixed sample/cluster settings are implied. Attention is classed as Static for all listed tasks, while State is Direct for prediction mappings and Constructed for QDG clustering.

## Evidence
### Task: next-token prediction (language modeling)
- "Consider the task of modeling the distribution of text on the internet." (Section 2)
- "We evaluate several models in the suite (ranging from 19m to 6.4b non-embedding parameters) on approximately 10 million tokens from the test set of The Pile. We record cross-entropy loss on every token..." (Section 4)
- Inference: `In Dynamics = Capped (inferred)` and `Attention Dynamic = Static (inferred)` are inferred from "a set of decoder-only transformers" operating on token contexts; `State Dynamic = Direct (inferred)` follows the glossary's reactive mapping framing for next-token prediction. `Out Dimension = 0D (inferred)` follows from per-sample next-token output. (Section 4; Section 5)

### Task: binary classification (multitask sparse parity)
- "The sparse parity prediction problem is simple: given a bit string of length n, compute the parity (sum mod 2) of a fixed subset of k of those bits." (Section 3.1)
- "Since answers are parities, this task can be treated as a binary classification problem on the subset of bit strings..." (Section 3.1)
- Inference: `In Dimension = 1D (t) (inferred)` is inferred from position-indexed bit strings; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` are inferred from a feedforward setup ("We train ReLU MLPs with a single hidden layer to solve this task..."). (Section 3.1; Section 3.2)

### Task: clustering (quanta discovery from gradients)
- "...we will attempt to cluster tokens in a language corpus according to what knowledge or skill LLMs use to predict those tokens from their context." (Section 5)
- "Quanta Discovery from Gradients (QDG): We will use spectral clustering on gradients to find clusters of samples whose gradient has nonzero cosine similarity." (Section 5)
- Inference: `In Dimension = 1D (t) (inferred)` comes from token-context samples; `In Dynamics = Capped (inferred)` and `Out Dynamics = Capped (inferred)` come from fixed sample/cluster settings in practice ("We cluster 10000 tokens..." and "n_clusters = 400"); `Attention Dynamic = Static (inferred)` is based on fixed affinity-matrix clustering; `State Dynamic = Constructed (inferred)` reflects explicit construction of gradient representations and affinity matrices (`A`, `C`, `\hat{C}`). (Section 5; Section 5.1; Appendix C.1)
