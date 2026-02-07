# Learning Modular Exponentiation with Transformers (Not specified in the paper.)
Source: Learning Modular Exponentiation with Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prediction (modular exponentiation) | tokens (base-B digits for a,b,c) (inferred) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | tokens (base-B digits for d) (inferred) | 1D (t) (inferred) | Fixed (inferred) |

## Summary
The paper studies a single prediction task: computing modular exponentiation results from tokenized integer inputs. Based on the base-B digit template and fixed operand range, the task appears to use fixed-length 1D token streams for both input and output, and the attention/state dynamics are inferred from the encoder–decoder transformer description.

## Evidence
### Task: prediction (modular exponentiation)
- "We train compact 4-layer encoder—decoder Transformers to predict d" (Abstract)
- "Modular exponentiation requires sampling integers a, b, c ∈ Z and outcome d ∈ Z such that a^b ≡ d mod c" (Section 3 Sample Generation and Model Training)
- "represent integers using base B digits in the template: V3 + a_1 ... a_n + b_1 ... b_n + c_1 ... c_n + d_1 ... d_n" (Section 3, Integer representations)
- Inference: Input/output as token sequences, 1D dimensions, fixed dynamics, static attention, and constructed state are inferred from the base-B digit template and the encoder–decoder transformer architecture with learned embeddings and attention heads.
