# Towards Efficient Neurally-Guided Program Induction for ARC-AGI (2024)
Source: Towards Efficient Neurally-Guided Program Induction for ARC-AGI.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Program induction (DSL program synthesis for ARC-AGI tasks) | Support input/output grid examples (Xs and Ys) | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Program token sequence / correct program P | 1D (t) (inferred) | Capped (inferred) |
| Similarity prediction for execution-guided search | Pair of grids (intermediate/current grid and target grid) | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Grid similarity score (0 to 1) | 0D (inferred) | Fixed (inferred) |
| Next-token transformation prediction with execution feedback | Intermediate or starting program state and target grid; decoded token prefix | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Probability distribution over next DSL token | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper’s primary coverage is program induction for ARC-AGI from 2D grid examples, with a token-sequence DSL program as the main output representation. It also includes an auxiliary grid-similarity prediction task used for execution-guided search in LGS, producing scalar similarity targets. Across the three paradigms, justified dimensions span 2D grid inputs, 1D token sequences, and 0D scalar outputs. Attention/state behavior ranges from static/direct mappings (similarity prediction) to dynamic/constructed decoding with explicit intermediate-state feedback in the proposed LTS setup.

## Evidence
### Task: Program induction (DSL program synthesis for ARC-AGI tasks)
- "The goal is to search (as efficiently as possible) for the program that solves each task in the test set, given the provided DSL." (Section The Problem)
- "More formally, given a DSL  $\Omega = \{\pi_1, \pi_2, ..., \pi_N\}$  containing N primitive functions denoted  $\pi_i$ , a search algorithm F(X,Y) given the support input examples  $X_s$  and support output examples  $Y_s$  must return within the allocated CPU and memory budgets a program P such that  $P(X_q) = Y_q$" (Section The Problem)
- "the solution consists of training a transformer to output a program, using a pre-determined grammar (DSL) and syntax, that solves the task." (Section Learning the Program Space)
- "A first full decoding loop is executed ... The full decoding loop stops until either the maximum sequence length (40, in our experiments) is reached" (Section The Search Algorithm)
- "this approach is not execution-guided, in that it does not receive feedback about the intermediate state of the program" (Section The Search Algorithm)
- Inference: `2D (x, y)` is inferred from repeated use of "input and output grid" examples; `1D (t)` output and `Capped` output dynamics are inferred from autoregressive token sequences with an explicit maximum sequence length of 40; `Static` attention and `Constructed` state are inferred from non-execution-guided decoding plus explicit construction of a probability distribution over program space.

### Task: Similarity prediction for execution-guided search
- "From there, it is possible to input two different grids and estimate their similarity." (Section Learning the Grid Space)
- "The experiments reported here used a Transformer encoder-only model with max pooling that outputs a flattened vector: a grid embedding." (Section Learning the Grid Space)
- "The training procedure consists of iteratively feeding a pair of grids into the model, retrieving their respective embeddings, and then penalizing deviations between their dot product and their ground truth distance." (Section Learning the Grid Space)
- "The number of transformations is converted to a similarity between zero and one." (Section Learning the Grid Space)
- Inference: `2D (x, y)` input is inferred because both inputs are grids; `0D` output and `Fixed` output dynamics are inferred because the target is a single similarity value in [0,1]; `Static` attention and `Direct` state are inferred because this task is a direct pair-to-score mapping without explicit runtime selection or maintained intermediate state in the model description.

### Task: Next-token transformation prediction with execution feedback
- "The concept is to train a model such that, given an intermediate or starting program state, and a target grid, it predicts the probability distribution over the DSL for the next token." (Section Learning the Transform Space)
- "the main difference with LPS is that we explicitly feed back into each decoding step some notion of the intermediate state of the program." (Section Learning the Transform Space)
- "there needs to be a hidden latent state maintained throughout the decoding process that gets updated at each step." (Section Learning the Transform Space)
- "LTS borrows the same program syntax and autoregressive sequence supervision as in GridCoder." (Section Learning the Transform Space)
- "This proxy experiment was done, instead of correctly training a model that learns to decode in such a way, due to lack of time." (Section Generalization of LTS)
- Inference: `2D (x, y); 1D (t)` input and `1D (t)` output are inferred from grid-state conditioning plus autoregressive token decoding; `Capped` dynamics are inferred from borrowing GridCoder’s autoregressive sequence setup with explicit sequence-length control; `Dynamic` attention and `Constructed` state are inferred from explicit per-step execution feedback and maintained latent intermediate state.
