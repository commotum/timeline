# Grammar as a Foreign Language (Not specified in the paper)
Source: Grammar as a Foreign Language.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generation (syntactic constituency parsing) | sentence tokens (words) | 1D (t) | Open (inferred) | Dynamic | Constructed (inferred) | linearized parse tree tokens (constituency labels and brackets) | 1D (t) | Open (inferred) |

## Summary
The paper applies a sequence-to-sequence model with attention to syntactic constituency parsing, mapping sentences to linearized parse-tree token sequences. The task operates over 1D token sequences for both inputs and outputs, and the interface supports variable-length sequences via end-of-sequence termination (so dynamics are marked Open, inferred). Attention is explicitly dynamic over encoder states, and the model maintains hidden and memory states, so the state dynamic is Constructed (inferred).

## Evidence
### Task: generation (syntactic constituency parsing)
- "Syntactic constituency parsing is a fundamental problem in natural language processing" (Abstract)
- "Syntactic constituency parsing can be formulated as a sequence-to-sequence problem if we linearize the parse tree" (Section 1 Introduction)
- "First, the network consumes the sentence in a left-to-right sweep" (Section 2.2 Linearizing Parsing Trees)
- "Then, it outputs the linearized parse tree" (Section 2.2 Linearizing Parsing Trees)
- "Every output sequence terminates with a special end-of-sequence token" (Section 2 LSTM+A Parsing Model)
- "define a distribution over sequences of variable lengths" (Section 2 LSTM+A Parsing Model)
- "uses an attention mechanism over the encoder LSTM states" (Section 2.1 Attention Mechanism)
- "creating vectors in memory" (Section 2.2 Linearizing Parsing Trees)
- Inference: In/Out Dynamics are marked Open because the model defines variable-length sequences with end-of-sequence termination and no explicit maximum length stated.
- Inference: State Dynamic is marked Constructed because the model explicitly creates and uses memory vectors during processing.
