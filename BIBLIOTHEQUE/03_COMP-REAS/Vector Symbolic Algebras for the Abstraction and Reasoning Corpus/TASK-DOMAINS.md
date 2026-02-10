# VECTOR SYMBOLIC ALGEBRAS FOR THE ABSTRACTION AND REASONING CORPUS (Not specified in the paper)
Source: Vector Symbolic Algebras for the Abstraction and Reasoning Corpus.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Few-shot abstract grid transformation generation/prediction (ARC-AGI-style 2D) | Demonstration pairs of input/output grids plus query input grids | 2D (x, y) | Capped | Dynamic (inferred) | Constructed (inferred) | Predicted query output grids (size and pixel contents) | 2D (x, y) | Capped |
| Few-shot abstract grid transformation generation/prediction (1D-ARC) | Demonstration pairs and query grids represented as single-row pixel grids | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Predicted single-row output grids | 1D (t) | Not specified in the paper. |

## Summary
The paper covers few-shot abstract grid-to-grid generation/prediction, centered on ARC-AGI-style 2D tasks and additionally evaluated on a 1D ARC variant. The supported dimensions are 2D (x, y) and 1D (t), with explicit capped variability in ARC-AGI-style grids, while 1D length dynamics are not explicitly bounded in the OCR text. The solver’s runtime behavior reflects Dynamic attention (inferred) and Constructed state (inferred) because it performs heuristic-guided hypothesis selection and builds explicit object/program abstractions.

## Evidence
### Task: Few-shot abstract grid transformation generation/prediction (ARC-AGI-style 2D)
- "ARC-AGI is a fluid intelligence benchmark comprising a collection of grid prediction tasks (see Figs. 1 and 2)." (Section 1 Introduction)
- "given a few pairs of input and output grids containing abstract symbols, determine the rules underlying the symbol transformations and use this understanding to predict the output grids corresponding to lone test input grids." (Section 1 Introduction)
- "Each grid,  $G \in \mathbb{G}$ , contains r rows and c columns of pixels." (Section 2.3.1 Background and Definition)
- "in ARC-AGI-1-Train,  $r, c \in \{1, \dots, 30\}$" (Section 2.3.1 Background and Definition)
- Inference: `Attention Dynamic = Dynamic` is inferred from runtime selection behavior: "our solver first tries the most promising hypotheses" and "our solver only considers actions applied to the input object most similar to the output object" (Section 3.2.1 Demonstration Abduction). `State Dynamic = Constructed` is inferred from explicit internal abstractions: "our solver represents each grid, input and output, as the set of its constituent objects" and "our solver produces a solution to each task as a program" (Sections 3.1.1 and 3.1.2).

### Task: Few-shot abstract grid transformation generation/prediction (1D-ARC)
- "1D-ARC (Xu et al., 2024) is a collection of 900 one-dimensional ARC-AGI-like tasks." (Section 4.2.2 1D-ARC)
- "all grids are a single row of pixels." (Section 4.2.2 1D-ARC)
- "Each task comprises  $|\mathcal{D}|=3$  demonstrations and  $|\mathcal{Q}|=1$  query" (Section 4.2.2 1D-ARC)
- Inference: `Attention Dynamic = Dynamic` and `State Dynamic = Constructed` are inferred from the same shared solver process used across benchmarks, including heuristic runtime selection and explicit object/program representations (Sections 3.1-3.2). `In Dynamics` and `Out Dynamics` remain "Not specified in the paper." because no explicit 1D grid-length bound is provided.
