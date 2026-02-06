# From Parrots to Von Neumanns: How Evolutionary Test-Time Compute Achieved State-of-the-Art on ARC-AGI (2025)
Source: From Parrots to Von Neumanns- How Evolutionary Test-Time Compute Achieved SOTA on ARC-AGI.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| transformation (ARC grid puzzle solving) | input-output grid pairs; test input grid | 2D (x, y) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | transformed grid (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |
| program synthesis (Python function generation) | training pairs (input-output grids) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Python functions | 1D (t) (inferred) | Not specified in the paper. |
| instruction generation (natural-language algorithm) | training examples (input-output grids) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | natural-language instructions | 1D (t) (inferred) | Not specified in the paper. |
| instruction execution (apply instructions to grids) | natural-language instructions; input grid | 1D (t) (inferred); 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | transformed grid | 2D (x, y) (inferred) | Not specified in the paper. |

## Summary
The paper centers on ARC visual puzzle solving, where a small set of example input-output grids is used to infer a transformation and produce a test output grid. It also explicitly covers program synthesis in Python and natural-language instruction generation, plus a follower model that executes instructions to produce grids. The supported dimensions include 2D grid-structured inputs/outputs and 1D text/code sequences (inferred), while dynamics, attention, and state are largely not specified beyond the capped number of training examples.

## Evidence
### Task: transformation (ARC grid puzzle solving)
- "Each task is a novel visual puzzle: given 2-4 inputoutput grid pairs, infer the transformation rule and apply it to a test input." (Section 1.1 The Riddle)
- Inference: Inferred 2D (x, y) inputs/outputs and capped input dynamics because the task is described as grid pairs with "2-4" examples and applying a rule to a test input. (Section 1.1 The Riddle)

### Task: program synthesis (Python function generation)
- "My first system (2024) evolved Python functions using Claude Sonnet 3.5, achieving 53.6% on ARC-AGI-1." (Abstract)
- "Require: Training pairs  $\mathcal{D} = \{(x_i, y_i)\}$ ; generator LLM G" (Algorithm 1)
- Inference: Inferred 2D (x, y) inputs because the training pairs are ARC grid pairs; inferred 1D (t) output because Python functions are code sequences. (Section 1.1; Algorithm 1)

### Task: instruction generation (natural-language algorithm)
- "Instead of Python functions, candidates are now structured natural-language instructions." (Section 5.2 The New Representation: Natural-Language Algorithms)
- "Planner. Generates initial instruction candidates. Given the training examples, proposes diverse strategies for solving the task." (Section 5.4 Role Specialization: Planner, Repairer, Follower)
- Inference: Inferred 2D (x, y) inputs because the training examples are ARC grid pairs; inferred 1D (t) output because the candidates are natural-language instructions. (Section 1.1; Section 5.2)

### Task: instruction execution (apply instructions to grids)
- "A separate \"follower\" LLM (also Grok-4 in my production system) reads these instructions and applies them to each grid." (Section 5.2 The New Representation: Natural-Language Algorithms)
- "Follower. Takes an instruction and a grid, outputs the transformed grid." (Section 5.4 Role Specialization: Planner, Repairer, Follower)
- Inference: Inferred 1D (t) for instructions and 2D (x, y) for grids because the inputs are textual instructions applied to grids. (Section 5.2; Section 5.4)
