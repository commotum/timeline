# Nested Learning: The Illusion of Deep Learning Architectures (Not specified in the paper.)
Source: Nested Learning- The Illusion of Deep Learning Architecture.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling | tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Common-sense reasoning | tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | labels (inferred) | 0D (inferred) | Not specified in the paper. |
| Continual learning | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Long-context reasoning | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper reports evaluations on language modeling and common-sense reasoning tasks, and it also claims results on continual learning and long-context reasoning. Input structure is only described at a high level via sequence modeling with token inputs, so 1D (t) dimensions are inferred for the language and common-sense tasks. Dynamics and attention behavior are not specified, while the architecture discussion suggests constructed internal state through context-compressing memory modules.

## Evidence
### Task: Language modeling
- "Performance of HOPE and baselines on language modeling and common-sense reasoning tasks." (Table 1)
- "showing promising results in language modeling, continual learning, and long-context reasoning tasks." (Abstract)
- "In sequence modeling, where keys and values are input tokens (e.g., tokenized text)" (Section 2.1)
- "lower perplexity and higher accuracy in benchmark results." (Section 4)
- "parameters  $\\boldsymbol{\\theta}_t^{(f_\\ell)}$  are responsible for compressing their own context into the their parameters" (Section 3)
- Inference: Input tokens and 1D (t) inferred from the sequence-modeling token statement; output tokens and 1D (t) inferred from language modeling with perplexity; State Dynamic marked Constructed from the context-compression statement.

### Task: Common-sense reasoning
- "Performance of HOPE and baselines on language modeling and common-sense reasoning tasks." (Table 1)
- "Language Modeling and Common-sense Reasoning." (Section 4)
- "In sequence modeling, where keys and values are input tokens (e.g., tokenized text)" (Section 2.1)
- "lower perplexity and higher accuracy in benchmark results." (Section 4)
- "parameters  $\\boldsymbol{\\theta}_t^{(f_\\ell)}$  are responsible for compressing their own context into the their parameters" (Section 3)
- Inference: Input tokens and 1D (t) inferred from the sequence-modeling token statement; output labels and 0D inferred from the use of accuracy on benchmark tasks; State Dynamic marked Constructed from the context-compression statement.

### Task: Continual learning
- "showing promising results in language modeling, continual learning, and long-context reasoning tasks." (Abstract)
- "continual learning abilities of HOPE" (Section 4)

### Task: Long-context reasoning
- "showing promising results in language modeling, continual learning, and long-context reasoning tasks." (Abstract)
- "long-context tasks" (Section 4)
