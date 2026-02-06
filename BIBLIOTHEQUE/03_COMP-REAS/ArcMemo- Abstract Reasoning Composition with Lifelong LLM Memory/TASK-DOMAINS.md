# ARCMEMO: ABSTRACT REASONING COMPOSITION WITH LIFELONG LLM MEMORY (Not specified in the paper.)
Source: ArcMemo- Abstract Reasoning Composition with Lifelong LLM Memory.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grid transformation prediction (ARC puzzle solving) | input-output example pixel grids and test input grids | 2D (x, y) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | output pixel grids | 2D (x, y) (inferred) | Not specified in the paper. |
| Program synthesis of grid transformation function | input-output example pixel grids | 2D (x, y) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | transformation function / code (text) | 1D (t) (inferred) | Not specified in the paper. |
| Concept abstraction (memory writing) | solution traces / reasoning steps (text) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | concept memory entries (natural language) | 1D (t) (inferred) | Not specified in the paper. |
| Memory selection (relevant concept retrieval) | problem descriptions and memory entries (text) | 1D (t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | top-k selected memory entries (text) | 1D (t) (inferred) | Capped (inferred) |
| Puzzle captioning (grid-to-text description) | ARC puzzle pixel grids | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | natural language puzzle captions/descriptions | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper covers ARC-AGI puzzle solving on 2D pixel grids, program synthesis of grid transformation functions, memory abstraction, memory selection, and grid-to-text captioning for preprocessing. Inputs and outputs span 2D grid-structured data and 1D text/code, with dynamics largely unspecified except for capped top-k retrieval in selection. The system uses dynamic attention and constructed state when selecting and integrating memory, while other steps are described without explicit attention/state constraints.

## Evidence
### Task: Grid transformation prediction (ARC puzzle solving)
- "Each ARC puzzle encodes a transformation rule that maps input to output pixel grids." (Section 4 Experiments, Benchmark Selection)
- "The objective of each puzzle is to infer its rule given several examples of input-output grid pairs, and produce the corresponding output grid." (Section 4 Experiments, Benchmark Selection)
- Inference: Labeled input/output as 2D (x, y) because puzzles are "pixel grids"; marked Dynamic/Constructed because "relevant concepts are selectively retrieved and integrated into the prompt." (Section 4 Experiments, Benchmark Selection; Abstract)

### Task: Program synthesis of grid transformation function
- "we instead use a program synthesis approach that queries for a transformation function to convert input to output grids." (Section 4 Experiments, Evaluation)
- "The code artifact provides more signal for reflection and also allows us to test proposed logic against reference pairs for feedback." (Section 4 Experiments, Evaluation)
- Inference: Labeled inputs as 2D (x, y) due to "pixel grids"; labeled output as 1D (t) because it is a "transformation function" code artifact; marked Dynamic/Constructed because "relevant concepts are selected and integrated into context." (Section 4 Experiments, Benchmark Selection; Section 4 Experiments, Evaluation; Introduction)

### Task: Concept abstraction (memory writing)
- "query a model to reflect on the solution trace and summarize specific general ideas" (Section 3.3 Memory Write: Concept Abstraction)
- "These reconstructed traces are then used to extract situation-suggestion pairs, forming structured memory entries for guiding future analyses." (Section 3.3 Memory Write: Concept Abstraction)
- Inference: Treated traces and entries as 1D (t) because concepts are "stored in natural language"; marked Constructed because the system forms "memory entries." (Abstract; Section 3.3 Memory Write: Concept Abstraction)

### Task: Memory selection (relevant concept retrieval)
- "introduce a selection mechanism only to include the most relevant subset of memory entries at problem-solving time." (Section 3.4 Memory Read: Concept Selection)
- "We then query a model for the top-k most relevant entries using the generated description." (Section 3.4 Memory Read: Concept Selection)
- Inference: Marked 1D (t) for inputs/outputs because selection operates on a generated "description" and textual memory entries; marked Dynamic/Constructed because it selects a "subset of memory entries"; marked Out Dynamics as Capped due to "top-k." (Section 3.4 Memory Read: Concept Selection)

### Task: Puzzle captioning (grid-to-text description)
- "we leverage a vision language model for this preprocessing step." (Section 3.4 Memory Read: Concept Selection)
- "We caption each puzzle using a structured prompt that separates concrete observations from speculative transformations." (Section 3.4 Memory Read: Concept Selection)
- Inference: Marked input as 2D (x, y) because ARC puzzles are "pixel grids"; marked output as 1D (t) because the system "caption[s] each puzzle." (Section 4 Experiments, Benchmark Selection; Section 3.4 Memory Read: Concept Selection)
