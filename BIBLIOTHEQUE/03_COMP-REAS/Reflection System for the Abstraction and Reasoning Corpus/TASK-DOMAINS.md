# Reflection System for the Abstraction and Reasoning Corpus (2025)
Source: Reflection System for the Abstraction and Reasoning Corpus.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARC grid transformation prediction | input-output example grid pairs and test input grid (2D matrices of numbers) | 2D (x, y) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | test output grid (2D matrix of numbers) | 2D (x, y) | Not specified in the paper. |
| Prediction selection (reflection model) | ARC task plus candidate predictions from multiple solvers (grids) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | selected final prediction grid | 2D (x, y) (inferred) | Not specified in the paper. |

## Summary
The paper centers on solving ARC tasks where systems infer transformations from example input-output grids and generate a test output grid, with grids represented as 2D matrices of numbers. It also introduces a reflection stage that selects the most likely correct prediction from multiple solvers for an ARC task. The paper does not specify interface dynamics, attention policy, or state construction details beyond these task descriptions.

## Evidence
### Task: ARC grid transformation prediction
- "The test-taker is provided with some input-output pairs as examples." (Figure 1)
- "apply it to the test input grid to obtain the test output grid." (Figure 1)
- "Each ARC grid is represented as a 2D matrix of numbers." (Section 2.2)

### Task: Prediction selection (reflection model)
- "the task and the prediction are presented to the reflection model, which chooses the correct final prediction." (Figure 3)
- "the reflection model processes all generated predictions from all the ARC solvers." (Section 4.2)
- Inference: Inputs/outputs are 2D grids because ARC tasks are represented as 2D matrices of numbers and the reflection model receives ARC tasks and solver predictions for those tasks. (Figure 3; Section 2.2)
