# Mini-ARC: Solving Abstraction and Reasoning Puzzles with Small Transformer Models (2024)
Source: Mini-ARC- Solving Abstraction and Reasoning Puzzles with Small Transformer Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARC puzzle solving (grid-to-grid transformation prediction) | 2D color grids (training input/output pairs + test input grid) | 2D (x, y) | Fixed | Static (inferred) | Direct (inferred) | 2D output color grid | 2D (x, y) | Fixed |

## Summary
The paper focuses on ARC puzzle solving where a model predicts an output color grid from multiple 2D input/output grid examples and a test input grid. The task operates over 2D grids with fixed 12x12 sizing and a fixed number of grids via padding. Attention is over a fixed sequence of tokens (static, inferred), and the system directly maps grids to outputs without external search or program synthesis (direct state, inferred).

## Evidence
### Task: ARC puzzle solving (grid-to-grid transformation prediction)
- "ARC puzzles are structured as a list of input/output grids" (Section "The Abstraction and Reasoning Corpus (ARC)")
- "The goal is to infer the transformation from the list of input/output grids and then apply the same transformation to a new input grid." (Section "The Abstraction and Reasoning Corpus (ARC)")
- "The benchmark is composed of 2D puzzles" (Section "The Abstraction and Reasoning Corpus (ARC)")
- "Mini-ARC-12 and Mini-ARC-v12 expect the input to be nine 12x12 grids" (Section "3.1.1 Input Representation")
- "All grids are padded to 12x12 using a padding token (0)" (Section "3.1.1 Input Representation")
- "all missing training pairs are padded with 12x12 grids as well." (Section "3.1.1 Input Representation")
- "a 12x12 output grid is added to the end" (Section "3.1.1 Input Representation")
- "final layer to project the output back to discrete colors" (Section "3.1 Model Architecture")
- "The full embedded sequence is passed through 16 Transformer encoder layers with self-attention mechanisms." (Section "3.1.3 Attention and Masking")
- "without the use of search, language models, or program synthesis." (Abstract)
- Inference: Attention is labeled Static because the model processes a fixed, fully present sequence with self-attention rather than selecting inputs at runtime. This is inferred from the fixed input construction and full-sequence self-attention. (Sections "3.1.1 Input Representation" and "3.1.3 Attention and Masking")
- Inference: State is labeled Direct because the system predicts output grids without external search or program synthesis; no constructed decision state beyond the provided grids is described. (Abstract)

---

## CSV Output (required)
CSV written to: /home/jake/Developer/timeline/BIBLIOTHEQUE/05_ML-FNDTNS/Mini-ARC- Solving Abstraction and Reasoning Puzzles with Small Transformer Models/.TASK-DOMAINS.csv.tmp.3a73f03f1a0542839afbb07f090539f5
