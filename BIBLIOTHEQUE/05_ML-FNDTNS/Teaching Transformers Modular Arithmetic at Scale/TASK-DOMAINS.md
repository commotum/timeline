# Teaching Transformers Modular Arithmetic at Scale (Not specified in the paper.)
Source: Teaching Transformers Modular Arithmetic at Scale.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prediction of modular sum (modular addition) | N-element integer sequence/vector, with elements in $\mathbb{Z}_q$ | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Single modular sum value $s \in \mathbb{Z}_q$ (sometimes represented as angular coordinates $(x,y)$) | 0D (inferred) | Fixed (inferred) |
| Prediction of other modular arithmetic functions ($h_{j,k}$) | N-element integer sequence in $\mathbb{Z}_q$ (position-dependent) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Single function output $h_{j,k}(a_1,\ldots,a_N) \mod q$ | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers two supervised arithmetic prediction tasks: modular addition and a class of non-addition modular arithmetic functions. In both cases, the model maps an input integer sequence to a single modular output value, which supports 1D (t) input and 0D output labeling. The interface is fixed per model setting (fixed N and q), with no runtime mechanism described for selecting variable context, so dynamics are Fixed and attention is Static (inferred). The tasks are framed as direct input-to-output mappings rather than agentic systems with maintained externalized task state, so state is Direct (inferred).

## Evidence
### Task: Prediction of modular sum (modular addition)
- "Given N elements  $[x_1,x_2...x_N], x_i \in \mathbb{Z}_q$ , compute  $s = \sum_{i=1}^N x_i \mod q$ ." (Section 1 Introduction)
- "Following prior work, we train models to add N elements mod q (fixed N and q for each model)." (Section 3 Methodology)
- "modular addition involves an an input sequence but a single output token" (Section 3.1 Proposed Improvements)
- Inference: `In Dimension = 1D (t)` is inferred from "input sequence"; `Out Dimension = 0D` is inferred from "single output token"; `In/Out Dynamics = Fixed` is inferred from "fixed N and q for each model"; `Attention Dynamic = Static` is inferred because the paper describes encoder-only transformer self-attention over the provided sequence without runtime retrieval/selection; `State Dynamic = Direct` is inferred because the task is posed as direct sequence-to-output prediction. (Sections 3, 3.1)

### Task: Prediction of other modular arithmetic functions ($h_{j,k}$)
- "Finally, we explore whether our methods enable ML models to learn other modular arithmetic functions beyond addition." (Section 6 Beyond Modular Addition)
- "We introduce a class of functions  $h:\mathbb{Z}_q^N\to\mathbb{Z}_q$  outside the aforementioned class" (Section 6 Beyond Modular Addition)
- "We train models to predict outputs from these functions, using the same setup as before..." (Section 6 Beyond Modular Addition)
- Inference: `In Dimension = 1D (t)` is inferred from sequence-form inputs and "these functions depend on input sequence positions"; `Out Dimension = 0D` is inferred from scalar modular function outputs in $\mathbb{Z}_q$; `In/Out Dynamics = Fixed`, `Attention Dynamic = Static`, and `State Dynamic = Direct` are inferred from reuse of the same fixed-N transformer prediction setup described for modular addition. (Sections 3, 6)
