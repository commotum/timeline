## 1. Basic Metadata

Title: "Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings".
Authors: "Yoav Gelberg<sup>1,2</sup>, Koshi Eguchi<sup>1</sup>, Takuya Akiba<sup>1</sup> and Edoardo Cetin<sup>1</sup> Sakana AI, <sup>2</sup>University of Oxford".
Year: Year not specified.
Venue: Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper proposes "Dropping the Positional Embeddings of LMs after training (DroPE)" to enable "zero-shot context extension" "without any long-context finetuning" for pretrained LMs beyond their pretraining sequence length (Introduction).

---

## 3. Tasks Evaluated

Task name: Language modeling (perplexity / loss).
Task type: Generation.
Dataset(s) used: "16B fineweb tokens".
Domain: Natural language text.
Quote: "Training loss curves for a RoPE and NoPE transformers on 16B fineweb tokens." (Figure 3)

Task name: Needle-in-a-haystack (NIAH) retrieval.
Task type: Other (retrieval / needle-in-a-haystack).
Dataset(s) used: "needle-in-a-haystack (NIAH) (Hsieh et al., 2024; Kamradt, 2023)".
Domain: Natural language text.
Quote: "we compare the perplexity and needle-in-a-haystack (NIAH) (Hsieh et al., 2024; Kamradt, 2023) performance" (Section 4).

Task name: RULER multi-query.
Task type: Other (retrieval).
Dataset(s) used: "RULER benchmark (Hsieh et al., 2024)".
Domain: Natural language text.
Quote: "multi-query: retrieve needles for several listed keys" (Section 5.1).

Task name: RULER multi-key.
Task type: Other (retrieval).
Dataset(s) used: "RULER benchmark (Hsieh et al., 2024)".
Domain: Natural language text.
Quote: "multi-key: retrieve the needle for one specified key" (Section 5.1).

Task name: RULER multi-value.
Task type: Other (retrieval).
Dataset(s) used: "RULER benchmark (Hsieh et al., 2024)".
Domain: Natural language text.
Quote: "multi-value: retrieve all needles for one key with a single query" (Section 5.1).

Task name: MultiFieldQA.
Task type: Other (long context language modeling task; type not specified).
Dataset(s) used: "LongBench (Bai et al., 2023)".
Domain: Natural language text.
Quote: "four different tasks from LongBench (Bai et al., 2023)" and "MultiFieldQA" (Section 5.1, Table 2).

Task name: MuSiQue.
Task type: Other (long context language modeling task; type not specified).
Dataset(s) used: "LongBench (Bai et al., 2023)".
Domain: Natural language text.
Quote: "four different tasks from LongBench (Bai et al., 2023)" and "MuSiQue" (Section 5.1, Table 2).

Task name: GovReport.
Task type: Other (long context language modeling task; type not specified).
Dataset(s) used: "LongBench (Bai et al., 2023)".
Domain: Natural language text.
Quote: "four different tasks from LongBench (Bai et al., 2023)" and "GovReport" (Section 5.1, Table 2).

Task name: LCC.
Task type: Other (long context language modeling task; type not specified).
Dataset(s) used: "LongBench (Bai et al., 2023)".
Domain: Natural language text.
Quote: "four different tasks from LongBench (Bai et al., 2023)" and "LCC" (Section 5.1, Table 2).

Task name: LM reasoning benchmarks (six tasks; names not specified).
Task type: Reasoning / relational.
Dataset(s) used: Not specified (benchmarks cited only).
Domain: Natural language text.
Quote: "six different LM reasoning benchmarks (Bisk et al., 2020; Clark et al., 2018; Mihaylov et al., 2018; Sakaguchi et al., 2021; Zellers et al., 2019)" (Section 5.1).

Task name: Standard LM benchmarks (names not specified).
Task type: Other (benchmark suite; task types not specified).
Dataset(s) used: Not specified.
Domain: Natural language text.
Quote: "Comparison of base SmollM with SmollM-DroPE on standard LM benchmarks" (Figure 9).

---

## 4. Domain and Modality Scope

- Single domain (natural language text): "language models (LM)" and "long context language modeling tasks" (Introduction, Table 2).
- Multiple domains within the same modality: Not stated; tasks are all language benchmarks such as "LongBench" and "RULER" (Section 5.1).
- Multiple modalities: Not stated; all evidence refers to language models and language benchmarks (Introduction; Section 5.1).
- Domain generalization or cross-domain transfer: Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Language modeling (perplexity / loss) | Not specified (single-task LM objective) | Not specified | Not specified | "train from scratch different LMs with half a billion parameters on 16B fineweb tokens" and "in-context validation perplexity" (Section 5.1) |
| Needle-in-a-haystack (NIAH) retrieval | Not specified | No (zero-shot context extension) | Not specified | "zero-shot context extension *without any long-context finetuning*" and "Zero-shot NIAH" (Introduction; Table 1) |
| RULER multi-query | Not specified | No (zero-shot context extension) | Not specified | "select three tasks from the RULER benchmark" and "multi-query" (Section 5.1) |
| RULER multi-key | Not specified | No (zero-shot context extension) | Not specified | "select three tasks from the RULER benchmark" and "multi-key" (Section 5.1) |
| RULER multi-value | Not specified | No (zero-shot context extension) | Not specified | "select three tasks from the RULER benchmark" and "multi-value" (Section 5.1) |
| MultiFieldQA | Not specified | No (zero-shot length generalization) | Not specified | "zeroshot length generalization on four different tasks from LongBench" and "MultiFieldQA" (Section 5.1; Table 2) |
| MuSiQue | Not specified | No (zero-shot length generalization) | Not specified | "zeroshot length generalization on four different tasks from LongBench" and "MuSiQue" (Section 5.1; Table 2) |
| GovReport | Not specified | No (zero-shot length generalization) | Not specified | "zeroshot length generalization on four different tasks from LongBench" and "GovReport" (Section 5.1; Table 2) |
| LCC | Not specified | No (zero-shot length generalization) | Not specified | "zeroshot length generalization on four different tasks from LongBench" and "LCC" (Section 5.1; Table 2) |
| LM reasoning benchmarks (six tasks; names not specified) | Not specified | Not specified | Not specified | "in-context performance across six different LM reasoning benchmarks" (Section 5.1) |
| Standard LM benchmarks (names not specified) | Not specified | Not specified | Not specified | "Comparison of base SmollM with SmollM-DroPE on standard LM benchmarks" (Figure 9) |

---

## 6. Input and Representation Constraints

- Sequence representation: "Let  h_1, \ldots, h_T \in \mathbb{R}^d  be the representations fed into a multi-head attention block." (Section 2)
- Sequence length is variable across training vs inference: "let the training and inference context lengths be  C_{\text{train}} < C_{\text{test}} , and define the extension factor  s = C_{\text{test}}/C_{\text{train}}" (Section 2).
- Explicit context length examples: "longer than 80 times SMOLLM's pretraining context (2048 tokens)" (Section 5.1).
- Fixed patch size: Not specified.
- Fixed number of tokens: Not specified beyond stated context lengths (e.g., "C_{\text{train}}" and "2048 tokens").
- Fixed dimensionality (e.g., strictly 2D): Not specified beyond "\mathbb{R}^d" token representations (Section 2).
- Padding or resizing requirements: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; evaluation mentions "2× the original context length" and "longer than 80 times SMOLLM's pretraining context (2048 tokens)" (Figure 1; Section 5.1).
- Sequence length fixed or variable: Variable, with "C_{\text{train}} < C_{\text{test}}" and extension factor "s = C_{\text{test}}/C_{\text{train}}" (Section 2).
- Attention type: Global (causal) self-attention, with "a  T \times T  matrix of attention scores" and softmax over "the first i tokens, implementing a causal mask" (Section 2).
- Mechanisms to manage computational cost: The paper notes attention is "inherently bottlenecked by quadratic token-to-token operations" but does not introduce windowing or pooling (Introduction).

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: RoPE is standard, and DroPE removes it. Evidence: "the modern literature has settled on the Rotary PE (RoPE) scheme" (Section 2) and "Dropping the Positional Embeddings of LMs after training (DroPE)" (Introduction).
- Where applied: RoPE is applied inside attention by rotating queries and keys, i.e., per layer/attention head: "providing relative positional information to each attention head by rotating  q_i  and  k_j  in 2D chunks before the inner product" (Section 2).
- Input only vs every layer vs attention bias: RoPE is applied to attention queries/keys in the attention computation: "RoPE modifies the attention scores ... by rotating queries and keys before taking their inner product" (Appendix A).
- Fixed vs modified vs compared: Positional encoding is varied and compared across experiments: "We repeat this recipe for RoPE and NoPE transformers, as well as an ALiBi model ... and an RNoPE-SWA model" and "We implement DroPE by taking the 14B tokens RoPE transformer checkpoint, removing positional embeddings from every layer" (Section 5.1). RoPE-scaling variants are also compared: "PI ... RoPE-NTK ... YaRN" (Section 2; Section 5.1).

---

## 9. Positional Encoding as a Variable

- Core research variable: Yes. The paper "challenge[s] the conventional role of RoPE in language modeling" and proposes "Dropping the Positional Embeddings" (Introduction).
- Multiple positional encodings compared: Yes. "We repeat this recipe for RoPE and NoPE transformers, as well as an ALiBi model ... and an RNoPE-SWA model" and also compare RoPE-scaling methods "PI ... RoPE-NTK ... YaRN" (Section 5.1; Section 2).
- PE choice described as not critical or secondary: Not stated.

---

## 10. Evidence of Constraint Masking

- Model sizes: "half a billion parameters" (Section 5.1), "360M parameter language model" (Section 5.1), and "sizes up to 7B parameters pretrained on trillions of tokens" (Introduction).
- Dataset sizes: "16B fineweb tokens" (Section 5.1), "pretrained on 600 billion tokens" (Section 5.1), and "trained on 1 trillion and 4 trillion tokens, respectively" (Section 5.1).
- Recalibration budgets / training tricks: "three different recalibration budgets of 30, 60, and 120 billion tokens" and "we also add QKNorm ... beneficial for mitigating training instabilities" (Section 5.1).
- Attribution of gains: Improvements are attributed to DroPE and removing positional embeddings rather than scaling alone: "DroPE yields seamless zero-shot context extension without any long-context finetuning" and "removing positional embeddings from every layer" (Introduction; Section 5.1).

---

## 11. Architectural Workarounds

- DroPE (drop PEs after pretraining) to enable long-context generalization: "Dropping the Positional Embeddings of LMs after training (DroPE)" and "removing positional embeddings from every layer" (Introduction; Section 5.1).
- RoPE frequency scaling for context extension (baseline methods): "PI ... RoPE-NTK ... YaRN define new RoPE frequencies" (Section 2).
- QKNorm for stability after dropping PEs: "we also add QKNorm after dropping the positional embeddings, which we find beneficial for mitigating training instabilities" (Section 5.1).

---

## 12. Explicit Limitations and Non-Claims

Not stated.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: single-modality language modeling and long-context NLP benchmarks.
> - Task structure: multiple long-context retrieval/QA/summarization benchmarks (RULER, LongBench, NIAH) plus LM benchmarks.
> - Representation rigidity: token sequences with explicit context length constraints (e.g., "C_{\text{train}} < C_{\text{test}}" and "2048 tokens").
> - Model sharing vs specialization: pretrained/recalibrated models evaluated across tasks with zero-shot claims; task-specific fine-tuning is not described.
> - Role of positional encoding: central experimental variable (RoPE vs NoPE vs DroPE and scaling methods).

---

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple NLP tasks and benchmarks, including "three tasks from the RULER benchmark" and "four different tasks from LongBench" as well as "six different LM reasoning benchmarks" (Section 5.1). All evaluations are within the language domain ("language models" and "long context language modeling tasks"), with no evidence of multiple modalities or cross-domain transfer (Introduction; Table 2).
