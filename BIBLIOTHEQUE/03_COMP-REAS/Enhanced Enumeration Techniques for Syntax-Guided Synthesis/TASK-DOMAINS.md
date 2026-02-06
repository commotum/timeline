# Enhanced Enumeration Techniques for Syntax-Guided Synthesis of Bit-Vector Manipulations (2024)
Source: Enhanced Enumeration Techniques for Syntax-Guided Synthesis.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bit-vector program synthesis (example-based SyGuS) | Input-output examples (bit-vector tuples); grammar/specification; natural language description (when provided) | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Bit-vector expression/program | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper defines an example-based syntax-guided synthesis task for bit-vector programs, consuming input-output examples (and, in some cases, a natural-language task description plus grammar) to produce a bit-vector expression/program. Based on the problem definition and implementation details, the inputs and outputs are 1D symbolic sequences with open-ended size, and the system can dynamically select subsets of examples during synthesis (inferred). The solver maintains internal search structures (e.g., term graphs and expression/condition sets), indicating constructed state during runtime (inferred).

## Evidence
### Task: Bit-vector program synthesis (example-based SyGuS)
- "S is a set of input-output examples" (Algorithm 1)
- "A solution to the SyGuS problem is an expression  $e \equiv \lambda x_1, \ldots, x_n. \gamma(x_1, \ldots, x_n)$" (Definition 3.5)
- "In addition to a collection of input-output examples that serve as an example-based specification, a natural language description of the synthesis task is also provided." (Example 4.9)
- Inference: In Dimension = 1D (t) and In Dynamics = Open because the input is a set of examples; Out Dimension = 1D (t) and Out Dynamics = Open because the output is an expression. Supporting text: "S is a set of input-output examples" (Algorithm 1); "A solution to the SyGuS problem is an expression  $e \equiv \lambda x_1, \ldots, x_n. \gamma(x_1, \ldots, x_n)$" (Definition 3.5).
- Inference: Attention Dynamic = Dynamic because "Dryadsynth takes a randomly selected subset of inputs from the specification S to be used in the synthesis process." (Section 6)
- Inference: State Dynamic = Constructed because "the algorithm maintains two sets: one for terms and another for predicates." (Section 4.2)
