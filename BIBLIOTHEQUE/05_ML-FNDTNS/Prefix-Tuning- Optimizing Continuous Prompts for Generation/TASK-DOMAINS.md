# Prefix-Tuning: Optimizing Continuous Prompts for Generation (Not specified in the paper.)
Source: Prefix-Tuning- Optimizing Continuous Prompts for Generation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| table-to-text generation | linearized data table | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | textual description | 1D (t) | Capped (inferred) |
| summarization | article | 1D (t) | Capped | Static (inferred) | Direct (inferred) | short summary | 1D (t) | Capped (inferred) |

## Summary
The paper evaluates prefix-tuning on two conditional text generation tasks: table-to-text generation and summarization, mapping linearized tables or articles to text outputs. Both inputs and outputs are textual sequences (1D (t)); summarization inputs are explicitly truncated to 512 BPE tokens and Transformer context is bounded, so dynamics are capped. Based on the autoregressive Transformer description, attention is treated as static and state as direct for these tasks (inferred).

## Evidence
### Task: table-to-text generation
- "We apply prefix-tuning to GPT-2 for table-to-text generation and to BART for summarization." (Abstract)
- "In table-to-text, x corresponds to a linearized data table and y is a textual description;" (§3 Problem Statement)
- "Transformers can only condition on a bounded-length context (e.g., 2048 tokens for GPT-3)" (§2 Prompting)
- "The autoregressive Transformer model computes  $h_i$  as a function of  $z_i$  and the past activations in its left context" (§3.1 Autoregressive LM)
- Inference: Labeled In/Out Dynamics as Capped and Attention/State as Static/Direct because the model uses a bounded context and computes each token from left-context activations.

### Task: summarization
- "We apply prefix-tuning to GPT-2 for table-to-text generation and to BART for summarization." (Abstract)
- "in summarization, x is an article and y is a short summary." (§3 Problem Statement)
- "the source articles are truncated to 512 BPE tokens." (§5.3 Architectures and Hyperparameters)
- "Transformers can only condition on a bounded-length context (e.g., 2048 tokens for GPT-3)" (§2 Prompting)
- "The autoregressive Transformer model computes  $h_i$  as a function of  $z_i$  and the past activations in its left context" (§3.1 Autoregressive LM)
- Inference: Labeled Out Dynamics as Capped and Attention/State as Static/Direct because the model uses a bounded context and computes each token from left-context activations.

---

## CSV Output (required)
task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic
table-to-text generation,linearized data table,1D (t),Capped (inferred),Static (inferred),Direct (inferred),textual description,1D (t),Capped (inferred)
summarization,article,1D (t),Capped,Static (inferred),Direct (inferred),short summary,1D (t),Capped (inferred)
