# NEURAL PROGRAM SEARCH: SOLVING PROGRAMMING TASKS FROM DESCRIPTION AND EXAMPLES (Not specified in the paper.)
Source: Solving Programming Tasks from Description and Examples.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Program synthesis (generation) from description and examples | Tokenized natural-language task description, argument signatures, and sample input/output tests | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | DSL program (tree-structured AST) | Not specified in the paper. | Capped (inferred) |

## Summary
The paper covers a single core task: synthesizing programs from natural-language descriptions with a small number of input/output examples. The input side is sequential text plus ordered example tests, which supports a 1D (t) input classification (inferred). The output is explicitly a tree-structured DSL program, but a coordinate-style output dimension label is not explicitly specified in the paper. The interface is treated as capped (inferred) because the paper uses bounded example usage and bounded search controls, with static attention over a predefined encoded input and constructed internal state via decoder/search structures.

## Evidence
### Task: Program synthesis (generation) from description and examples
- "We present a Neural Program Search, an algorithm to generate programs from natural language description and a small number of input / output examples." (Abstract)
- "We specifically consider a problem of synthesizing programs from a short description and several input / output pairs." (Section 1 Introduction)
- "The encoder uses RNN to embed concatenation of arguments Args and tokenized textual description of the task Text." (Section 3.2 SeQ2Tree)
- "The decoder is a doubly-recurrent neural network for generating tree structured output" (Section 3.2 SeQ2Tree)
- "The search continues until a complete program is found that passes given sample input / output pairs." (Section 3.3 Search)
- Inference: In Dimension is labeled 1D (t) because the paper explicitly uses "tokenized textual description" and ordered test cases; In Dynamics is labeled Capped from "a small number" of examples and the explicit inference setup "using first 3 tests for search" (Section 4). Attention Dynamic is labeled Static because attention is applied over a predefined encoded input statement at each decode step (Section 3.2). State Dynamic is labeled Constructed because the decoder combines parent/sibling hidden states and search maintains incomplete-tree structures in a priority queue (Section 3.2, Section 3.3). Out Dynamics is labeled Capped due explicit bounded search controls such as "MAX_VISITED" and "QUEUE_N" (Algorithm 1, Section 3.3).
