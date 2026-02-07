# Latent Execution for Neural Program Synthesis (Not specified in the paper)
Source: Latent Execution for Neural Program Synthesis (LaSynth).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Program synthesis (restricted C list-processing) | 5 input-output pairs of numerical lists | 1D (t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Restricted C program tokens | 1D (t) (inferred) | Capped |
| Program synthesis (Karel grid-world) | 5 input-output pairs of Karel grid world states | 2D (x, y) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Karel program tokens | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper studies program synthesis from input-output examples in two domains: restricted C list-processing and Karel grid-world programs. Inputs are IO pairs of numerical lists or 2D grid states, and outputs are program token sequences; only the C domain explicitly caps program length at 256 tokens. The architecture uses attention over IO pairs/tokens and maintains latent execution traces, indicating dynamic attention and constructed state for both tasks.

## Evidence
### Task: Program synthesis (restricted C list-processing)
- "we make the first attempt of synthesizing C code in a restricted domain from input-output examples only, and we focus on programs for list processing." (Section 2)
- "we randomly sample 5 numerical lists as the program inputs" (Section 4.1)
- "execute the program to obtain the corresponding output lists." (Section 4.1)
- "The goal of the program synthesizer is to predict a program P from  $\{IO\}^K$" (Section 2)
- "we only constrain the final program length ( $\leq 256$  tokens)" (Section 4.1)
- "the next program token  $p_t$" (Section 3.1)
- "we compute an attention vector  $d_t$  over previously generated program tokens" (Section 3.1)
- "which maintains a second representation  $\hat{I}_t$  during program decoding." (Section 3.2)
- Inference: In Dimension = 1D (t) from "numerical lists"; Out Dimension = 1D (t) from "program token"; Attention Dynamic = Dynamic and State Dynamic = Constructed from the attention and latent executor descriptions above. (Section 4.1; Section 3.1; Section 3.2)

### Task: Program synthesis (Karel grid-world)
- "A Karel program controls a robot in a 2D grid world." (Section 5.1.1)
- "Each program includes 5 input-output pairs as the specification, and the sixth pair as the held-out test case." (Section 5.1.1)
- "The goal of the program synthesizer is to predict a program P from  $\{IO\}^K$" (Section 2)
- "the next program token  $p_t$" (Section 3.1)
- "we compute an attention vector  $d_t$  over previously generated program tokens" (Section 3.1)
- "which maintains a second representation  $\hat{I}_t$  during program decoding." (Section 3.2)
- Inference: Out Dimension = 1D (t) from "program token"; Attention Dynamic = Dynamic and State Dynamic = Constructed from the attention and latent executor descriptions above. (Section 3.1; Section 3.2)
