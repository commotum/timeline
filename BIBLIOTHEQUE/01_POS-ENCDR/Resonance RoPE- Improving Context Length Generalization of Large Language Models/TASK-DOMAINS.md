# Resonance RoPE: Improving Context Length Generalization of Large Language Models (Not specified in the paper)
Source: Resonance RoPE- Improving Context Length Generalization of Large Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Recursive next-token generation (POSGEN) | Token sequences `{x_0, ..., x_{l-1}}` with local dependency | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Generated next tokens / token sequences | 1D (t) (inferred) | Capped (inferred) |
| Chain-of-Thought next-token generation (POSGEN) | Token sequences `{x_0, ..., x_{l-1}}` with front-and-local dependency | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Generated next tokens / token sequences | 1D (t) (inferred) | Capped (inferred) |
| Semi-recursive next-token generation (POSGEN) | Token sequences `{x_0, ..., x_{l-1}}` with varied-distance dependency | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Generated next tokens / token sequences | 1D (t) (inferred) | Capped (inferred) |
| Long-text language modeling | Long-text token sequences / text fragments | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Next-token probabilities or predictions (evaluated by perplexity) | 1D (t) (inferred) | Capped (inferred) |
| Long-text close-ended task answering (L-Eval) | Long-text prompts/documents from L-Eval domains | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Close-ended answers | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers five task rows: three POSGEN synthetic next-token generation subtasks (Recursive, CoT, Semi-recursive), long-text language modeling, and real-world long-text close-ended task answering. All described inputs are text/token sequences, supporting 1D (t) input dimensionality, and experiment interfaces are capped by explicit sequence/context limits (e.g., 64/256 and 4K to 32K or 16K). Based on the described Transformer setup and next-token objective, attention is classified as Static (inferred) and state as Direct (inferred). Outputs span token-sequence predictions (1D) and close-ended answers (0D inferred).

## Evidence
### Task: Recursive next-token generation (POSGEN)
- "We consider a next token prediction task, where we expect the model to generate the token  $x_l$  given the input sequence  $\{x_0,\cdots,x_{l-1}\}$ ." (Section 5)
- "**Recursive.** This task simulates the token dependency pattern of generating a Fibonaccistyle sequence, where new tokens depend on j+k neighboring tokens only:" (Section 5)
- "The models are trained on sequences of length L=64, and evaluating on lengths of L' = 256 for OOD Accuracy." (Section 6.1.1)
- Inference: `1D (t)` is inferred from token-sequence indexing over positions `l`; `Capped` is inferred from explicit train/test length bounds (`L=64`, `L'=256`); `Static` attention is inferred from standard Transformer self-attention (Section 3.1: "In Transformers ... the self-attention scores are softmax-normalized scaled attention logits"); `Direct` state is inferred because the task is explicit next-token prediction from the provided sequence.

### Task: Chain-of-Thought next-token generation (POSGEN)
- "We consider a next token prediction task, where we expect the model to generate the token  $x_l$  given the input sequence  $\{x_0,\cdots,x_{l-1}\}$ ." (Section 5)
- "Chain-of-Thought (CoT). This task simulates the token dependency pattern of CoT reasoning (Wei et al., 2022), where new tokens depend on k neighboring tokens (simulating the previous reasoning step) and j tokens in the front (simulating the original question):" (Section 5)
- "The models are trained on sequences of length L=64, and evaluating on lengths of L' = 256 for OOD Accuracy." (Section 6.1.1)
- Inference: `1D (t)` is inferred from token-sequence indexing over positions `l`; `Capped` is inferred from explicit train/test length bounds (`L=64`, `L'=256`); `Static` attention is inferred from standard Transformer self-attention (Section 3.1); `Direct` state is inferred because the task remains next-token prediction from sequence context.

### Task: Semi-recursive next-token generation (POSGEN)
- "We consider a next token prediction task, where we expect the model to generate the token  $x_l$  given the input sequence  $\{x_0,\cdots,x_{l-1}\}$ ." (Section 5)
- "**Semi-recursive.** This task simulates the token dependency pattern of the last-letter concatenation task (Zhou et al., 2023), where new tokens depend on both k neighboring tokens (simulating the current progress) and j tokens with varied distances according to a specific rule (simulating the word sequence):" (Section 5)
- "The models are trained on sequences of length L=64, and evaluating on lengths of L' = 256 for OOD Accuracy." (Section 6.1.1)
- Inference: `1D (t)` is inferred from token-sequence indexing over positions `l`; `Capped` is inferred from explicit train/test length bounds (`L=64`, `L'=256`); `Static` attention is inferred from standard Transformer self-attention (Section 3.1); `Direct` state is inferred because the task remains next-token prediction from sequence context.

### Task: Long-text language modeling
- "We test the model's performance on two TSTL scenarios: language modeling evaluation on long-text sequences and long-text downstream application performance." (Section 6.2.1)
- "We evaluate the model's language modeling performance on GovReport (Huang et al., 2021) and Proofpile (Azerbayev, 2022)." (Section 6.2.2)
- "For YaRN and RESONANCE YARN, We use a scaling factor of 8 and 4 for LLaMA2 7B and 13B to extend their context window from 4K to 32K and 16K, respectively." (Section 6.2.1)
- Inference: `1D (t)` is inferred from long text/token sequences; `Capped` is inferred from explicit context-window caps (32K/16K); `Static` attention is inferred from the Transformer self-attention setup (Section 3.1); `Direct` state is inferred from language-modeling behavior and objective (Appendix C.2: "The model was trained with a language modeling-style cross entropy loss"); output dimensionality `1D (t)` is inferred because predictions are made over token sequences.

### Task: Long-text close-ended task answering (L-Eval)
- "Lastly, we test the real-world task performance of LLaMA2-Chat 7B and 13B's performance with different RoPE scaling strategies on L-Eval (An et al., 2023)'s close ended task suite, a long-text LLM benchmark covering a wide range of domains such as school lectures, long conversations and novels." (Section 6.2.3)
- "Table 2: Long text evaluations on some closed-ended tasks in L-Eval." (Table 2 caption)
- "For YaRN and RESONANCE YARN, We use a scaling factor of 8 and 4 for LLaMA2 7B and 13B to extend their context window from 4K to 32K and 16K, respectively." (Section 6.2.1)
- Inference: `1D (t)` input is inferred from long-text benchmark descriptions; `Capped` input dynamics is inferred from explicit context-window caps; `Static` attention is inferred from the same Transformer self-attention formulation; `Direct` state is inferred because no constructed external state/memory mechanism is described; output as `Close-ended answers`, with `0D` and `Fixed`, is inferred from the paper’s repeated "close ended/closed-ended" framing of this task suite.
