# Beyond Brute Force: A Neuro-Symbolic Architecture for Compositional Reasoning in ARC-AGI-2 (2025)
Source: Beyond Brute Force- A Neuro-Symbolic Architecture for Compositional Reasoning in ARC-AGI-2.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid-to-grid transformation prediction (ARC-AGI-2 puzzles) | input-output grid examples and test input grid (color grids) | 2D (x, y) | Capped (inferred) | Static | Constructed | output grid (final solved grid) | 2D (x, y) | Capped (inferred) |

## Summary
The paper focuses on ARC-AGI-2 visual reasoning tasks that require predicting an output grid from input-output grid examples and a test input grid. The modality is 2D color grids with bounded size, implying capped spatial dynamics. The system uses a fixed set of provided grids as input (static attention) while constructing symbolic scene graphs and rule hints as internal state.

## Evidence
### Task: grid-to-grid transformation prediction (ARC-AGI-2 puzzles)
- "This revamped benchmark maintains the same input—output grid format and core-knowledge constraints" (Section 1 Introduction)
- "Each task consists of a small number of input-output grid examples" (Section 2.1 ARC-AGI-1)
- "All training input/output grid pairs." (Section 3.4 Stage 4: LLM Solving with Self-Consistency)
- "Output only the final solved grid in the specified format." (Section 3.4 Stage 4: LLM Solving with Self-Consistency)
- "Converts the raw  $N \times M$  pixel grids into a structured scene graph of symbolic objects and their properties." (Section 3 Our Approach: ARC-AGI Compositional Reasoning)
- Inference: Set In/Out Dynamics to "Capped" because the paper states "Grids up to  $20\times20$  cells," indicating a bounded grid size. (Section 2.2.3 Challenges of Grid Scale)

---

## CSV Output (required)
task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic
grid-to-grid transformation prediction (ARC-AGI-2 puzzles),input-output grid examples and test input grid (color grids),2D (x, y),Capped (inferred),Static,Constructed,output grid (final solved grid),2D (x, y),Capped (inferred)
