# The LLM ARChitect: Solving ARC-AGI Is A Matter of Perspective (2024)
Source: The LLM ARChitect- Solving ARC-AGI Is a Matter of Perspective.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARC-AGI grid transformation generation | Paired ARC task grids (X_i, Y_i) plus unseen input grid X^c | 2D (x, y) | Capped | Static (inferred) | Constructed (inferred) | Predicted output grid candidates for Y^c | 2D (x, y) | Capped |
| Candidate solution scoring and selection (classification) | Task grid context and candidate output grids across augmentations | 2D (x, y) | Capped | Static (inferred) | Constructed (inferred) | Candidate probabilities/ranking used to choose two guesses | 0D | Capped |

## Summary
The paper covers ARC-AGI colored-grid reasoning with two distinct task intents for the model: generating output grids and classifying/ranking candidate solutions. The task objects are grid-structured and 2D, with capped dynamics supported by explicit grid-size limits (1x1 to 30x30) and finite inference/selection constraints. The outputs include both 2D grid predictions and 0D probability/ranking signals. Attention and state are not explicitly labeled in the paper, but the described decoder-only inference and DFS/scoring procedures support Static attention and Constructed state (both inferred).

## Evidence
### Task: ARC-AGI grid transformation generation
- "Each task involves grids of varying sizes, ranging from 1x1 to 30x30, utilizing a palette of ten distinct colors." (Section 3.1 Datasets)
- "Each instance consists of two grids: one representing the input of the problem and the other representing the expected output. The objective is to infer the underlying mechanics from a few examples and apply this understanding to a new, unseen instance as illustrated in the figure." (Section 3.1 Datasets)
- "To generate a solution candidate we repeat this procedure until we sample an  $\langle eos \rangle$  token, indicating that the example is done. The output is parsed into an array, with some checks to make sure it is a valid grid." (Section 3.5 Solution Inference)
- Inference: Attention Dynamic is Static (inferred) because generation is described as autoregressive next-token prediction from provided tokens ("When provided with tokens  $x_1, \ldots, x_n$ , a model M will calculate the probability distributions  $p_{x_2}, \ldots, p_{x_{n+1}}$  for subsequent token predictions." in Section 3.5) without runtime selection of external observations. State Dynamic is Constructed (inferred) because the method builds and traverses a search structure with reusable caches ("We employ a **depth-first search** (**DFS**) to explore all possible paths through the solution tree... By leveraging inference caches..." in Section 3.5).

### Task: Candidate solution scoring and selection (classification)
- "leveraging our generative model both as a predictor and as a classifier for good solutions." (Abstract)
- "Finally, given the generated list of candidates, we use the aggregated logsoftmax scores assigned by the fine-tuned model to select two of them for submission." (Section 2 Pipeline Overview, Candidate Selection)
- "Given a task C and a solution candidate  $S_k$ , the model can calculate a probability  $P_M(S_k|C)$ , which represents how likely it is that the model would generate  $S_k$  when provided with C using standard sampling." (Section 3.6 Selection Strategies)
- "using no more than two guesses." (Section 3.6 Selection Strategies)
- Inference: Attention Dynamic is Static (inferred) because scoring conditions on fixed task/candidate inputs and predefined augmentations, without a described runtime mechanism that selects new observations. State Dynamic is Constructed (inferred) because the system computes and aggregates augmented probabilities as explicit decision state ("We therefore calculate the augmented probabilities ... for augmentations  $T_1, \ldots, T_n$ ... taking the product of the probabilities ... led to a highly stable and effective selection strategy." in Section 3.6).
