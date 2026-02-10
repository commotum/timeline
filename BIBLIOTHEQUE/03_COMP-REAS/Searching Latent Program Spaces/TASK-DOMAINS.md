# Searching Latent Program Spaces (Not specified in the paper)
Source: Searching Latent Program Spaces.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Program synthesis generalization on ARC-AGI grids | Input-output example pairs of 2D color grids plus a new input grid | 2D (x, y) | Capped | Static (inferred) | Constructed (inferred) | Predicted output 2D color grid | 2D (x, y) | Capped |
| Pattern grid transformation from examples | Input-output example pairs of 10x10 grids (blue marker in black grid) plus a new input grid | 2D (x, y) | Fixed | Static (inferred) | Constructed (inferred) | 10x10 output grid with pasted 4x4 pattern | 2D (x, y) | Fixed |
| Sequence/string manipulation from examples | Input-output example pairs of integer sequences plus a new input sequence | 1D (t) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Transformed integer sequence | 1D (t) | Not specified in the paper. |

## Summary
The paper covers programming-by-examples tasks in two modality domains: 2D color grids (ARC-AGI and Pattern) and 1D integer sequences (string manipulation). For grids, the OCR supports both Fixed dynamics (Pattern: always 10x10) and Capped dynamics (ARC-style grids up to 30x30). Sequence length bounds are not explicitly stated, so sequence dynamics are not specified in the paper. Across tasks, the model constructs latent program state and performs latent optimization at test time; attention control is inferred as static over the provided specification.

## Evidence
### Task: Program synthesis generalization on ARC-AGI grids
- "predict the corresponding output for  $x_{n+1}^m$ ." (Section 3, Program Synthesis Generalization)
- "input-output pairs represented as 2D grid of shape up to 30x30" (Section 3, Program Synthesis Generalization)
- "programs are defined in the input-output space of ARC-AGI, i.e., 2D grids" (Section G)
- Inference: `Static` attention is inferred from optimization over all provided pairs ("$\sum_{i=1}^{n} \log p_{\theta}(y_i|x_i, z)$", Section 4.2) rather than runtime retrieval/selection; `Constructed` state is inferred from "an explicit representation of programs via a compact latent space" and latent refinement (Section 4).

### Task: Pattern grid transformation from examples
- "It generates 10x10 black input grids with a blue pixel" (Section 5.1, Setup)
- "This specific task always generates fully-black 10x10 inputs" (Section A.1, Pattern Task)
- "the output pastes a 4x4 pattern" (Section A.1, Pattern Task)
- Inference: `Static` attention and `Constructed` state are inferred from the shared LPN inference procedure: fixed specification pairs are aggregated and optimized in latent space before decoding (Sections 4, 4.2, and Algorithm 1).

### Task: Sequence/string manipulation from examples
- "a synthetic sequence task" (Section 5.3, String Manipulation Task)
- "transform sequences of numbers (ranging from 0 to 4)" (Section 5.3, String Manipulation Task)
- "we process the sequence from left to right" (Section B.7.1, Dataset)
- Inference: Output as transformed sequences is supported by rule-based sequence transformation text (Section B.7.1). `Static` attention and `Constructed` state are inferred by applying the same LPN inference mechanism described in Sections 4 and C. Sequence length bounds are not explicitly given, so In/Out Dynamics are marked "Not specified in the paper."
