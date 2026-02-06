# A Neurosymbolic Approach to Abstraction and Reasoning (2021)
Source: A Neurosymbolic Approach to Abstraction and Reasoning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grid transformation / visual reasoning (ARC task solving) | grid examples (input/output pairs; test input grids) | 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | output grid(s) for test examples | 2D (x, y) (inferred) | Not specified in the paper. |
| Arithmetic expression synthesis (24 Game) | four input numbers and target 24 | 1D (t) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | arithmetic expression that creates 24 | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper covers two task domains: ARC-style few-shot visual reasoning where systems map input grids to output grids, and 24 Game puzzle solving via arithmetic expression synthesis. The ARC task domain operates over 2D grid inputs/outputs, while the 24 Game operates over a fixed-size set of numbers and generates arithmetic expressions; output dynamics are not explicitly bounded. The described synthesis framework maintains an explicit search-state graph and selects operations/arguments at runtime, implying constructed state and dynamic attention for these tasks.

## Evidence
### Task: Grid transformation / visual reasoning (ARC task solving)
- "Each training example is an input/output pair of grids." (Section 2 The Abstraction and Reasoning Corpus)
- "produce the correct output grid for each of the test examples" (Section 2 The Abstraction and Reasoning Corpus)
- "Each task consists of a 2–4 training examples and one or more test examples." (Section 2 The Abstraction and Reasoning Corpus)
- "The current state is a graph of nodes." (Section 4.3 Bidirectional, Execution-guided Program Synthesis)
- "chooses one of O operations to apply and selects M arguments for the operation" (Section 4.3 Network and training)
- Inference: In/Out Dimension are 2D (x, y) because the task uses "grids"; In Dynamics is Capped because tasks have "2–4 training examples"; Attention is Dynamic and State is Constructed because the system "chooses" operations/arguments at runtime and maintains a "graph of nodes" with intermediate values.

### Task: Arithmetic expression synthesis (24 Game)
- "A 24 Game consists of four input numbers and a target number 24." (Section 4.4 Results)
- "use each number once in an arithmetic expression that creates 24." (Section 4.4 Results)
- "The current state is a graph of nodes." (Section 4.3 Bidirectional, Execution-guided Program Synthesis)
- "chooses one of O operations to apply and selects M arguments for the operation" (Section 4.3 Network and training)
- Inference: In Dimension is 1D (t) and In Dynamics is Fixed because the task specifies "four input numbers"; Out Dimension is 1D (t) because the output is an "arithmetic expression"; Attention is Dynamic and State is Constructed because the system "chooses" operations/arguments at runtime and maintains a "graph of nodes" with intermediate values.
