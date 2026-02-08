# TransXSSM: A Hybrid Transformer–State Space Model with Unified Rotary Position Embedding (Year not specified in the paper)
Source: TransXSSM- Hybrid Transformer–SSM with Unified RoPE.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling | Input tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Next-token probabilities / tokens | 1D (t) (inferred) | Capped (inferred) |
| MMLU benchmark prediction | Prompt tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer label (inferred) | 0D (inferred) | Capped (inferred) |
| TriviaQA benchmark prediction | Prompt tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer text (inferred) | 1D (t) (inferred) | Capped (inferred) |
| ARC benchmark prediction | Prompt tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer label (inferred) | 0D (inferred) | Capped (inferred) |
| PIQA benchmark prediction | Prompt tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer label (inferred) | 0D (inferred) | Capped (inferred) |
| HellaSwag benchmark prediction | Prompt tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer label (inferred) | 0D (inferred) | Capped (inferred) |
| OBQA benchmark prediction | Prompt tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer label (inferred) | 0D (inferred) | Capped (inferred) |
| Winogrande benchmark prediction | Prompt tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer label (inferred) | 0D (inferred) | Capped (inferred) |
| Long-context retrieval (needle-in-a-haystack) | Long document with embedded sentence | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Retrieved sentence (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper covers text-sequence tasks: language modeling, seven named downstream language benchmarks, and long-context retrieval. The supported task space is predominantly 1D (t) token/document sequences, with output dimension split between 0D label outputs (ACC-reported tasks) and 1D (t) text outputs (QEM/retrieval), where noted as inferred. Reported sequence limits (4K, 8192, and up to 16K) support Capped dynamics, and the causal masked setup supports Static attention with Direct state (inferred).

## Evidence
### Task: Language modeling
- "it surpasses a Transformer baseline by over 4% on language modeling benchmarks." (Abstract)
- "linear language model head to predict the next-token probabilities." (Section 3)
- Inference: 1D (t), Capped, Static, and Direct are inferred from token-sequence processing and fixed context bounds: "At a 4K sequence length" and "contexts up to 16K tokens," plus causal masking in Section 2.1.

### Task: MMLU benchmark prediction
- "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (Section B.1)
- "MMLU ACC ↑" (Table 3)
- Inference: Prompt tokens, 1D (t), Capped, Static, and Direct are inferred from the language-task framing, causal language-model architecture, and reported sequence/context limits; "ACC" supports a 0D label output inference.

### Task: TriviaQA benchmark prediction
- "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (Section B.1)
- "pure State-Space models (Mamba2) are strong in direct knowledge extraction (e.g., TriviaQA)." (Section 4.2)
- Inference: Prompt tokens, 1D (t), Capped, Static, and Direct are inferred as above; "TRIVIAQA QEM ↑" in Table 3 supports an answer-text (1D (t)) output inference.

### Task: ARC benchmark prediction
- "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (Section B.1)
- "pure Transformers (LlaMa3) excel in directed reasoning (e.g., ARC)" (Section 4.2)
- Inference: Prompt tokens, 1D (t), Capped, Static, and Direct are inferred from the same language-model setup; benchmark "ACC" reporting supports a 0D label output inference.

### Task: PIQA benchmark prediction
- "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (Section B.1)
- "tasks requiring commonsense reasoning and contextual understanding (e.g., HellaSwag, PIQA, Winogrande)." (Section 4.2)
- Inference: Prompt tokens, 1D (t), Capped, Static, and Direct are inferred from the same language-model setup; benchmark "ACC" reporting supports a 0D label output inference.

### Task: HellaSwag benchmark prediction
- "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (Section B.1)
- "tasks requiring commonsense reasoning and contextual understanding (e.g., HellaSwag, PIQA, Winogrande)." (Section 4.2)
- Inference: Prompt tokens, 1D (t), Capped, Static, and Direct are inferred from the same language-model setup; benchmark "ACC" reporting supports a 0D label output inference.

### Task: OBQA benchmark prediction
- "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (Section B.1)
- "OBQA ACC↑" (Table 3)
- Inference: Prompt tokens, 1D (t), Capped, Static, and Direct are inferred from the same language-model setup; "ACC" supports a 0D label output inference.

### Task: Winogrande benchmark prediction
- "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (Section B.1)
- "On Winogrande, the TransXSSM-1.3B outperforms its LlaMa3 counterpart by nearly 6.7 points" (Section 4.2)
- Inference: Prompt tokens, 1D (t), Capped, Static, and Direct are inferred from the same language-model setup; benchmark "ACC" reporting supports a 0D label output inference.

### Task: Long-context retrieval (needle-in-a-haystack)
- "in a challenging long-context \"needle-in-a-haystack\" retrieval task" (Section 1)
- "embedding a \"needle\" (random sentence) in a \"haystack\" (long document) for retrieval." (Section 4.2)
- Inference: 1D (t), Capped, Static, and Direct are inferred because the task is described as long-document language-sequence retrieval under the same causal, bounded-context model setup.
