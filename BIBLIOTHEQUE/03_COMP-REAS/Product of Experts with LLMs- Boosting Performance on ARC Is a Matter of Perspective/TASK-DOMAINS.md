# Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective (2025)
Source: Product of Experts with LLMs- Boosting Performance on ARC Is a Matter of Perspective.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid-to-grid transformation (ARC-AGI/ConceptARC) | input-output example grids and a test input grid | 2D (x, y) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | output grid | 2D (x, y) (inferred) | Capped (inferred) |
| Sudoku puzzle solving | Sudoku puzzles | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | correct Sudoku solution | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper applies its method to ARC-AGI and ConceptARC, which are grid-to-grid transformation puzzles defined over bounded-size grids, and it also evaluates on Sudoku puzzles. The ARC-like tasks justify 2D grid inputs/outputs with capped size, while the paper does not specify attention or state dynamics for the systems. For Sudoku, the paper mentions puzzles and correct solutions but does not spell out the input/output dimensionality or dynamics.

## Evidence
### Task: grid-to-grid transformation (ARC-AGI/ConceptARC)
- "Each task involves input and output grids of varying sizes, ranging from 1x1 to 30x30 and utilize a palette of ten distinct colors." (Section: The Original ARC Dataset)
- "discern the transformation rule from input to output from the examples and apply it to new input grids to generate the correct outputs." (Section: The Original ARC Dataset)
- "a small set of k inputoutput examples and a single test input." (Section: Problem Representation)
- "ConceptARC (Moskvichev et al., 2023) - an ARC-like dataset containing tasks sorted into specific conceptual categories." (Section: 5.5. ConceptARC)
- Inference: Labeled inputs/outputs as 2D (x, y) with Capped dynamics because tasks are described as "input and output grids" that range from "1x1 to 30x30".

### Task: Sudoku puzzle solving
- "We further test our approach on the Sudoku 3M dataset (Radcliffe, 2020) to evaluate generalizability of the method to different domains." (Section: 5.6. Sudoku)
- "This setup reaches 53% accuracy on 1000 randomly chosen unseen Sudoku puzzles" (Section: 5.6. Sudoku)
- "if the correct solution of a puzzle is sampled, we select it in 100% of cases." (Section: 5.6. Sudoku)
