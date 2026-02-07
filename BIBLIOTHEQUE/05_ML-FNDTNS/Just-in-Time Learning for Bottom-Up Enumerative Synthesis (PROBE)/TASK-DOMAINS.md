# Just-in-Time Learning for Bottom-Up Enumerative Synthesis (2020)
Source: Just-in-Time Learning for Bottom-Up Enumerative Synthesis (PROBE).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| program synthesis (string manipulation) | context-free grammar (DSL) and input-output examples (strings) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Constructed | program from the DSL that satisfies the examples | Not specified in the paper. | Not specified in the paper. |
| program synthesis (bit-vector manipulation) | context-free grammar (DSL) and universally-quantified first-order formula specification (BitVec) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Constructed | program satisfying the specification | Not specified in the paper. | Not specified in the paper. |
| program synthesis (circuit transformation) | context-free grammar (DSL) and universally-quantified boolean formula specification (Circuit) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Constructed | circuit/program satisfying the specification | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper describes a SyGuS program synthesizer that generates DSL programs from specifications, evaluated on string manipulation, bit-vector manipulation, and circuit transformation domains. String tasks use input-output example specifications, while BitVec and Circuit tasks use universally-quantified first-order or boolean formulas (handled via standard SyGuS formulations). The algorithm explicitly maintains a constructed search state (program bank, caches, partial solutions), but it does not specify input/output dimensionality, dynamics, or attention behavior.

## Evidence
### Task: program synthesis (string manipulation)
- "We evaluate Probe on three different application domains: string manipulation (String), bit-vector manipulation (BitVec), and circuit transformations (Circuit)." (Sec. 6.1 Experimental Setup)
- "All these benchmarks use input-output examples as semantic specification" (Sec. 6.1 Experimental Setup)
- "The tool takes as input an inductive synthesis problem in SyGuS format, i.e. a context-free grammar of the DSL and a set of input-output examples" (Introduction, The Probe tool)
- "it outputs a program from the DSL that satisfies all the examples." (Introduction, The Probe tool)
- "The algorithm maintains a search state that consists of (1) the current cost level Lvl; (2) program bank B" (Sec. 4.2 Guided Bottom-up Search Algorithm)

### Task: program synthesis (bit-vector manipulation)
- "We evaluate Probe on three different application domains: string manipulation (String), bit-vector manipulation (BitVec), and circuit transformations (Circuit)." (Sec. 6.1 Experimental Setup)
- "The input to a SYGUS problem is a syntactic specification, in the form of a context-free grammar (CFG) that defines the space of possible programs" (Sec. 2.1 Syntax Guided Synthesis)
- "The semantic specification of BitVec benchmarks is a universally-quantified first-order formula that is functionally equivalent to the target program." (Sec. 6.1 Experimental Setup)
- "A *solution* to the problem is a program  $P \in \mathcal{L}(\mathcal{G})$" (Sec. 4.1 Preliminaries)
- "The algorithm maintains a search state that consists of (1) the current cost level Lvl; (2) program bank B" (Sec. 4.2 Guided Bottom-up Search Algorithm)

### Task: program synthesis (circuit transformation)
- "We evaluate Probe on three different application domains: string manipulation (String), bit-vector manipulation (BitVec), and circuit transformations (Circuit)." (Sec. 6.1 Experimental Setup)
- "The input to a SYGUS problem is a syntactic specification, in the form of a context-free grammar (CFG) that defines the space of possible programs" (Sec. 2.1 Syntax Guided Synthesis)
- "These benchmarks involve synthesizing constant-time circuits that are cryptographically resilient to timing attacks." (Sec. 6.1 Experimental Setup)
- "The semantic specification is a universally-quantified boolean formula functionally equivalent to the circuit to be synthesized." (Sec. 6.1 Experimental Setup)
- "The algorithm maintains a search state that consists of (1) the current cost level Lvl; (2) program bank B" (Sec. 4.2 Guided Bottom-up Search Algorithm)
