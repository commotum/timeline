# EXTENDING CONTEXT WINDOW OF LARGE LANGUAGE MODELS VIA POSITION INTERPOLATION (Not specified in the paper.)
Source: Extending Context Window of Large Language Models via Positional Interpolation (PI).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Long sequence language modeling | tokens (text) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (next token prediction) | 1D (t) (inferred) | Capped (inferred) |
| Passkey retrieval | tokens (long document with embedded passkey) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (passkey digits) | 1D (t) (inferred) | Capped (inferred) |
| Long document summarization | tokens (long document) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (summary) | 1D (t) (inferred) | Capped (inferred) |
| BoolQ benchmark evaluation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| PIQA benchmark evaluation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Race-M benchmark evaluation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Race-H benchmark evaluation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| WinoGrande benchmark evaluation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper evaluates text-based tasks that require long context, specifically long sequence language modeling, passkey retrieval, and long document summarization, and also reports results on benchmarks (BoolQ, PIQA, Race-M/H, WinoGrande). The described tasks operate on token sequences and produce token outputs with explicit caps from context-window limits and truncation settings. Based on the fixed context windows and unmodified attention, the attention and state dynamics are inferred as static and direct.

## Evidence
### Task: Long sequence language modeling
- "We evaluate the long sequence language modeling performance of our extended models and baselines" (Section 3.2 Long Sequence Language Modeling)
- "We fine-tune all model variants using the next token prediction objective." (Section 3.1 Setup)
- Inference: Inferred `1D (t)` and `Capped` from "inputs to LLaMA models (Touvron et al., 2023) must be fewer than 2048 tokens." (Section 1 Introduction). Inferred `Static` from "Our work allows full access of the entire input through unmodified attention" (Section 4 Related Work). Inferred `Direct` from "We fine-tune all model variants using the next token prediction objective." (Section 3.1 Setup)

### Task: Passkey retrieval
- "we follow a synthetic evaluation task of passkey retrieval proposed by Mohtashami & Jaggi (2023)." (Section 3.3 Measuring Effective Context Window Size through Passkey Retrieval)
- "the models are asked to recover a random passkey hidden in a long document." (Section 3.3 Measuring Effective Context Window Size through Passkey Retrieval)
- "What is the pass key? The pass key is" (Figure 3)
- Inference: Inferred `1D (t)` and `Capped` from "inputs to LLaMA models (Touvron et al., 2023) must be fewer than 2048 tokens." (Section 1 Introduction) and the long-document prompt in Figure 3. Inferred `Static` from "Our work allows full access of the entire input through unmodified attention" (Section 4 Related Work). Inferred `Direct` from the prompt-style completion "What is the pass key? The pass key is" (Figure 3).

### Task: Long document summarization
- "In this task, we evaluate our models' performance on the long document summarization task." (Section 3.5 Long Document Summarization)
- "Each document comes with a human generated summary." (Section 3.5 Long Document Summarization)
- "We truncate all input documents to their first 15000 tokens." (Section 3.5 Long Document Summarization)
- "The final output is truncated at 1000 tokens." (Section 3.5 Long Document Summarization)
- "Read the following article and then summarize it." (Figure 4)
- Inference: Inferred `1D (t)` and `Capped` from "We truncate all input documents to their first 15000 tokens." and "The final output is truncated at 1000 tokens." (Section 3.5 Long Document Summarization). Inferred `Static` from "Our work allows full access of the entire input through unmodified attention" (Section 4 Related Work). Inferred `Direct` from "We fine-tune the model using the next token prediction task" (Section 3.5 Long Document Summarization)

### Task: BoolQ benchmark evaluation
- "We evaluate the models extended by Position Interpolation on several standard benchmark tasks within the original context window size of 2048." (Section 3.4 BENCHMARKS ON ORIGINAL CONTEXT WINDOW SIZE)
- "BoolQ | PIQA | Race-M | Race-H | WinoGrande" (Table 5)

### Task: PIQA benchmark evaluation
- "We evaluate the models extended by Position Interpolation on several standard benchmark tasks within the original context window size of 2048." (Section 3.4 BENCHMARKS ON ORIGINAL CONTEXT WINDOW SIZE)
- "BoolQ | PIQA | Race-M | Race-H | WinoGrande" (Table 5)

### Task: Race-M benchmark evaluation
- "We evaluate the models extended by Position Interpolation on several standard benchmark tasks within the original context window size of 2048." (Section 3.4 BENCHMARKS ON ORIGINAL CONTEXT WINDOW SIZE)
- "BoolQ | PIQA | Race-M | Race-H | WinoGrande" (Table 5)

### Task: Race-H benchmark evaluation
- "We evaluate the models extended by Position Interpolation on several standard benchmark tasks within the original context window size of 2048." (Section 3.4 BENCHMARKS ON ORIGINAL CONTEXT WINDOW SIZE)
- "BoolQ | PIQA | Race-M | Race-H | WinoGrande" (Table 5)

### Task: WinoGrande benchmark evaluation
- "We evaluate the models extended by Position Interpolation on several standard benchmark tasks within the original context window size of 2048." (Section 3.4 BENCHMARKS ON ORIGINAL CONTEXT WINDOW SIZE)
- "BoolQ | PIQA | Race-M | Race-H | WinoGrande" (Table 5)
