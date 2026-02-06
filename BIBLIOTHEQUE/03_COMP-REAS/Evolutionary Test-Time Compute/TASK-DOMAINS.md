# How I came in first on ARC-AGI-Pub using Sonnet 3.5 with Evolutionary Test-time Compute (2024)
Source: Evolutionary Test-Time Compute.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid transformation | input/output example grids; test input grid | 2D (x, y) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | output grid | 2D (x, y) | Not specified in the paper. |
| program synthesis (Python transform function) | input/output example grids | 2D (x, y) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Python transform function (code) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
Across the paper, the system addresses ARC-AGI puzzles that require transforming example grids into a test output grid, i.e., 2D grid-to-grid reasoning. The method uses an LLM to synthesize Python transform functions from grid examples, introducing a separate code-generation task with 1D token outputs. Attention and state are inferred as dynamic/constructed due to iterative selection and evolution of functions, while input/output dynamics (fixed/capped/open) are not specified.

## Evidence
### Task: grid transformation
- "Here, you are given two examples of input/output grids and you must fill in the test output grid with the correct colors" (Section "What is ARC")
- "convert input grids to output grids" (Section "Architecture")
- Inference: Attention Dynamic and State Dynamic are inferred because the method selects parents and iterates generations ("The best-performing functions are selected as parents"; "This process repeats, with each generation of functions typically performing"). (Section "Architecture")

### Task: program synthesis (Python transform function)
- "My approach works by having Sonnet 3.5 generate a bunch of Python transform functions" (Introduction)
- "I have the LLM generate Python functions, instead of just outputting solution grids" (Introduction)
- Inference: Attention Dynamic and State Dynamic are inferred because the method selects parents and iterates generations ("The best-performing functions are selected as parents"; "This process repeats, with each generation of functions typically performing"). Out Dimension is inferred as 1D (t) because outputs are Python functions ("I have the LLM generate Python functions, instead of just outputting solution grids"). (Section "Architecture"; Introduction)

## CSV Output (required)
