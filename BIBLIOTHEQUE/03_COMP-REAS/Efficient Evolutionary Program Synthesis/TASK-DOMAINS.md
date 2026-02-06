# Efficient Evolutionary Program Synthesis (2025)
Source: Efficient Evolutionary Program Synthesis.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| program synthesis | input/output grids of colored cells (training examples) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | Python program(s) | 1D (t) (inferred) | Not specified in the paper. |
| grid transformation (ARC-AGI task solving) | input/output grids of colored cells (training examples) and test input grids | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | output grids of colored cells | 2D (x, y) (inferred) | Not specified in the paper. |

## Summary
The paper describes ARC-AGI tasks where systems infer rules from training input/output grids and apply them to test inputs to produce output grids. It also presents an LLM-assisted program synthesis system that generates Python programs to solve those grid tasks while building a program library across tasks. Inputs and grid outputs are 2D (inferred), program outputs are 1D token sequences (inferred), dynamics and attention are not specified, and state is constructed via the evolving library (inferred).

## Evidence
### Task: program synthesis
- "Each ARC task has several training examples in the form of input/output grids of colored cells which encode some unwritten rules." (Section: Background, ARC-AGI)
- "Starting from an empty library, my system loops through each task to prompt an LLM for Python program(s) that can solve all of the training examples." (Section: Architecture)
- "growing system expertise by adding promising programs to a library" (Section: Motivation)
- Inference: Treated inputs as 2D (x, y) because they are "grids of colored cells"; treated output programs as 1D (t) because they are "Python program(s)"; treated state as constructed because the system grows a program library. (Section: Background, ARC-AGI; Section: Architecture; Section: Motivation)

### Task: grid transformation (ARC-AGI task solving)
- "Each ARC task has several training examples in the form of input/output grids of colored cells which encode some unwritten rules." (Section: Background, ARC-AGI)
- "The goal of an AI system is discovering those rules, and then applying them to test inputs to generate output grids." (Section: Background, ARC-AGI)
- "growing system expertise by adding promising programs to a library" (Section: Motivation)
- Inference: Treated inputs and outputs as 2D (x, y) because they are "grids" of colored cells and output grids; treated state as constructed because the system grows a program library. (Section: Background, ARC-AGI; Section: Motivation)
