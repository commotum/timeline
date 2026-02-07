# How Can Self-Attention Networks Recognize Dyck-n Languages? (Not specified in the paper)
Source: How Can Self-Attention Networks Recognize Dyck-n Languages-.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multi-label next-symbol prediction (Dyck-n recognition) | Dyck-n bracket symbol sequence with starting symbol T | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Set of next valid bracket symbols (k-hot labels) per position | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper frames Dyck-n language recognition as an auto-regressive transduction that predicts the set of valid next brackets at each position in a bracket-symbol string. Inputs and outputs are bracket-symbol sequences, implying a 1D (t) structure, and the experimental setup uses bounded sequence lengths, so dynamics are capped (both inferred). The models use suffix-masked self-attention and are described as memory-less, supporting static attention and direct state (both inferred).

## Evidence
### Task: Multi-label next-symbol prediction (Dyck-n recognition)
- "recognition of  $\mathcal{D}_n$  languages as a transduction task" (Section 3 Experiments)
- "Given a valid string, we ask the model to predict the next possible symbols auto-regressively." (Section 3 Experiments)
- "input vocabulary  $(V_n^i)$  for a  $\mathcal{D}_n$  language consists of 2n+1 symbols" (Section 3 Experiments)
- "n pairs of brackets (or parentheses), and an additional starting symbol T" (Section 3 Experiments)
- "output vocabulary  $(V_n^o)$  does not include the starting symbol T." (Section 3 Experiments)
- "outputs are encoded as a k-hot vector" (Section 3 Experiments)
- "train on 32k sequences of length 2-50" (Section 3 Experiments)
- "evaluate on 10k sequences divided equally over the length intervals 76-100 and 102-126." (Section 3 Experiments)
- "Softmax attention scores of the second layer of a suffix-masked  $SA^+$" (Figure 1 caption)
- "The ability of (memory-less) SA networks to recognize  $\mathcal{D}_{n>1}$  languages is intriguing." (Section 4 Compatibility With a Stack-Based Recognizer)
- Inference: In/Out Dimension labeled 1D (t) (inferred) from the use of strings and per-position prediction; In/Out Dynamics labeled Capped (inferred) from bounded length ranges; Attention Dynamic labeled Static (inferred) from suffix-masked self-attention over a fixed prefix; State Dynamic labeled Direct (inferred) from the description of memory-less SA networks.
