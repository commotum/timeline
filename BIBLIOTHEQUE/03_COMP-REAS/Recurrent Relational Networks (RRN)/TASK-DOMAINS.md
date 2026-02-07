# Recurrent Relational Networks (Not specified in the paper.)
Source: Recurrent Relational Networks (RRN).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| question answering (text) | facts (short sentences) + question (sentence) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | single-word answer (bAbI vocabulary) | 0D (inferred) | Fixed (inferred) |
| question answering (scene) | scene with eight colored shapes + question (start object + jumps) | 2D (x, y) (inferred); 0D (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | shape or color (answer) | 0D (inferred) | Fixed (inferred) |
| puzzle solving (Sudoku completion) | 9x9 Sudoku grid with digits (givens) + row/column position | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | digit for each cell (solution grid) | 2D (x, y) (inferred) | Fixed (inferred) |
| age arithmetic (age inference) | one absolute age + set of age differences (statements) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | age (number) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers textual question answering (bAbI), relational question answering over 2D scenes (Pretty-CLEVR), Sudoku puzzle completion on a 9x9 grid, and an age-arithmetic inference task. Inputs span 1D text sequences and 2D spatial/grid structures with mostly fixed or capped sizes, while outputs are single-word/label answers or full-grid digit predictions (0D or 2D). Attention is described over fixed input sets (static, inferred) and the RRN uses recurrent hidden states (constructed, inferred) where architecture details are provided, while the age-arithmetic setup omits architectural specifics.

## Evidence
### Task: question answering (text)
- "bAbI is a text based QA dataset from Facebook [Weston et al., 2015]" (Section 3.1)
- "is preceded by a number of facts in the form of short sentences" (Section 3.1)
- "The target is a single word" (Section 3.1)
- "up to a maximum of the last 20 facts." (Section 3.1)
- "At each step t each node has a hidden state vector h_i^t" (Section 2)
- Inference: In Dimension = 1D (t), Out Dimension = 0D, and Out Dynamics = Fixed because the input is sentences and the target is a single word; In Dynamics = Capped because inputs use a "maximum of the last 20 facts"; Attention Dynamic = Static from the fixed fact window; State Dynamic = Constructed from the recurrent hidden state quoted above.

### Task: question answering (scene)
- "Pretty-CLEVR consists of scenes with eight colored shapes and associated questions." (Section 3.2)
- "If the start object is defined by color, the answer is a shape, and vice versa." (Section 3.2)
- "We consider each scene as a fully connected undirected graph with 8 nodes." (Section 3.2)
- "The feature vector for each object consists of the position, shape and color." (Section 3.2)
- "At each step t each node has a hidden state vector h_i^t" (Section 2)
- Inference: In Dimension = 2D (x, y) and 0D because inputs include object positions plus a discrete question; In Dynamics = Fixed because each scene has 8 nodes; Attention Dynamic = Static from the fixed scene size; State Dynamic = Constructed from the recurrent hidden state quoted above.

### Task: puzzle solving (Sudoku completion)
- "We consider each of the 81 cells in the 9x9 Sudoku grid a node in a graph" (Section 3.3)
- "takes as input the digit for the cell (0-9, 0 if not given), and the row and column position (1-9)." (Section 3.3)
- "maps each node hidden state to nine output logits corresponding to the nine possible digits." (Section 3.3)
- "At each step t each node has a hidden state vector h_i^t" (Section 2)
- Inference: In/Out Dimension = 2D (x, y) and In/Out Dynamics = Fixed because the task uses a 9x9 grid with 81 cells and digit outputs per cell; Attention Dynamic = Static from the fixed grid; State Dynamic = Constructed from the recurrent hidden state quoted above.

### Task: age arithmetic (age inference)
- "The task is to infer the age of a person given a single absolute age and a set of age differences" (Section 3.4)
- Inference: In Dimension = 1D (t) because the inputs are text statements; Out Dimension = 0D and Out Dynamics = Fixed because the task asks for a single age value.
