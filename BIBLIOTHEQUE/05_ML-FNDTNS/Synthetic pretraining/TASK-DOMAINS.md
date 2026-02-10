# Synthetic pretraining (Not specified in the paper)
Source: Synthetic pretraining.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Conversational and creative writing generation | Tokens/prompts (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Text responses (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Common-sense and multi-hop reasoning | Tokens/statements (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | Reasoning answers in text (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Document classification | Documents/text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Topic/class assignment (inferred) | 0D (inferred) | Not specified in the paper. |
| JSON generation | Tokens/prompts (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | JSON structures (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Mathematical proof generation and geometry problem solving | Unsolved math problems and optional proved lemmas | 1D (t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Proofs and candidate conjectures | 1D (t) (inferred) | Not specified in the paper. |
| Programming and agentic coding | Programming environments, Issues/PRs, and code artifacts | 1D (t) (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | Code/task solutions (inferred) | 1D (t) (inferred) | Open (inferred) |
| Agentic search and tool-use orchestration | Problems plus tool-call results (inferred) | 1D (t) (inferred) | Open (inferred) | Dynamic | Constructed (inferred) | Tool-call/action trajectories with reasoning steps (inferred) | 1D (t) (inferred) | Open (inferred) |

## Summary
The OCR text describes a broad LLM-centric coverage spanning text generation, reasoning, classification, formal math proving, and agentic coding/search workflows. The explicitly named tasks are mostly language-structured and therefore map to 1D (t) in/out dimensions (inferred), with one classification output mapped to 0D (inferred). Dynamics are usually not explicitly specified by interface constraints, but agentic/tool-driven workflows support an Open characterization (inferred). Attention is only clearly dynamic in tool-use orchestration; state is repeatedly described in ways consistent with constructed internal reasoning primitives for reasoning-heavy tasks.

## Evidence
### Task: Conversational and creative writing generation
- "there will always be some frequency of conversational data, classification tasks, creative writing" (p. 0, Section "What is synthetic pretraining?")
- "less well, a range of generative behaviors." (p. 0, Section "What is synthetic pretraining?")
- Inference: Input/output were mapped to token sequences and `1D (t)` because the task is described as conversational/creative text behavior in an LLM context.

### Task: Common-sense and multi-hop reasoning
- "achieves common sense reasoning benchmark results comparable to models ten times its size" (p. 0, Section "What is synthetic pretraining?")
- "all model architectures fail at simplest 2-hop reasoning" exercises (p. 0, Section "What is synthetic pretraining?")
- "hop-reasoning and other logical constructs should be simply learned throughout training." (p. 0, Section "What is synthetic pretraining?")
- Inference: `Constructed` state was inferred from the paper's emphasis on learning reusable "reasoning primitives" and "logical constructs" rather than only direct pattern matching.

### Task: Document classification
- "Closest experiments in this direction come from Nemotron-CC where document classification served to reinforce \"correlations between advanced topics that are otherwise rarely observed in web-scale data\"." (p. 0, Section "Synthetic compilation 2: logic and pipelines")
- "you have to imagine a SYNTH-like dataset no longer derived from Wikipedia but from Wikidata, structured input from interconnected knowledge graphs." (p. 0, Section "Synthetic compilation 2: logic and pipelines")
- Inference: Input was mapped to textual documents (`1D (t)`), and output was mapped to `0D` class/topic assignments because the paper states the task as document classification but does not spell out the label schema.

### Task: JSON generation
- "is JSON generation, as you can simply filter out all faulty structure to only keep perfect examples in the final set." (p. 0, Section "What is synthetic pretraining?")
- "to design structured input and output that will come in handy once the model is plugged into integrated pipelines." (p. 0, Section "What is synthetic pretraining?")
- Inference: Input/output were mapped to token-structured sequences (`1D (t)`) because JSON is presented as generated structured text.

### Task: Mathematical proof generation and geometry problem solving
- "synthetic datasets allowing us to generate proofs for each of these domains at will while controlling precisely by how many proofs we augment our training sequence" (p. 0, Section "Synthetic compilation 2: logic and pipelines")
- "a proposer module accepts an unsolved problem and, optionally, some already proved lemmas as input, and generates 10-50 candidate conjectures about properties of the problem" (p. 0, Section "Synthetic compilation 2: logic and pipelines")
- "problem generation program found over 230 million unique problems" (p. 0, Section "Synthetic compilation 2: logic and pipelines")
- Inference: `Dynamic` attention and `Constructed` state were inferred from the described orchestration/backtracking over lemmas and candidate conjectures, implying adaptive consideration and maintained intermediate reasoning structures.

### Task: Programming and agentic coding
- "six specialized domains: mathematics, programming, general logical reasoning, general agentic tasks, agentic coding, and agentic search" (p. 0, Section "What is synthetic pretraining?")
- "Minimax M2.1 was trained on 100,000 programming environments with a constant recursive synthetic feedback" (p. 0, Section "Synthetic compilation 2: logic and pipelines")
- "automatically discovering high-quality Issues and PRs from GitHub; using models to assess task difficulty and perform stratification" (p. 0, Section "Synthetic compilation 2: logic and pipelines")
- Inference: `Open` dynamics were inferred from the recursive environment/task loop, and `Constructed` state from repeated stratification/augmentation over evolving task context.

### Task: Agentic search and tool-use orchestration
- "GLM 4.5 was trained on \"large-scale synthetic agent trajectories\"." (p. 0, Section "Synthetic compilation 3: simulations")
- "interleaved thinking", that is the process "which enables Claude about the results of a tool call before deciding what to do next" and "chain multiple tool calls with reasoning steps in between". (p. 0, Section "Synthetic compilation 3: simulations")
- "parallel agent orchestration, allowing the model to decide it should tackle a problem through parallel rather than sequential search." (p. 0, Section "Synthetic compilation 3: simulations")
- Inference: `Open` in/out dynamics and `Constructed` state were inferred because the paper describes ongoing multi-step trajectories and intermediate reasoning between tool calls; `Dynamic` attention is directly supported by deciding what to do next from tool results.
