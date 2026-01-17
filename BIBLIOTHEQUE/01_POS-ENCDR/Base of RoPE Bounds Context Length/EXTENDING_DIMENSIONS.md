## 1. Basic Metadata

Title: "Base of RoPE Bounds Context Length" (page 0)
Authors: Xin Men; Mingyu Xu; Bingning Wang; Qingyu Zhang; Hongyu Lin; Xianpei Han; Weipeng Chen. Evidence: "Xin Men\*" "Mingyu Xu\*" "Bingning Wang\*" "Qingyu Zhang Hongyu Lin Xianpei Han" "Weipeng Chen" (page 0)
Year: Year not specified.
Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

The paper claims it "derive[s] that the base of RoPE bounds context length: there is an absolute lower bound for the base value to obtain certain context length capability" and validates the relationship between RoPE base and context length theoretically and empirically (Abstract).

## 3. Tasks Evaluated

- Task name: Perplexity (language modeling). Task type: Generation. Dataset(s): PG19 test split. Domain: Language modeling / text. Quotes: "For evaluation, we test the long context capabilities comprehensively, the benchmarks are listed below: **perplexity** on PG19 (Rae et al., 2019) test split. We evaluate the perplexity of each sample and get the mean value across samples." (Appendix B); "For attention mechanism in language modeling" (Section 4).
- Task name: Long-eval. Task type: Other (long-context retrieval QA). Dataset(s): Long-eval (Li* et al., 2023). Domain: Language modeling / text. Quote: "**Long-eval** (Li\* et al., 2023). This test generates massive random similar sentences and asks the model to answer questions according to a specific sentence in the context." (Appendix B).
- Task name: Retrieval. Task type: Other (retrieval). Dataset(s): Retrieval (Mohtashami & Jaggi, 2024). Domain: Language modeling / text. Quote: "Retrieval (Mohtashami & Jaggi, 2024)" (Appendix B).
- Task name: Needle in Haystack (NIH). Task type: Other (long-context retrieval). Dataset(s): Needle in Haystack (G, 2023). Domain: Language modeling / text. Quote: "**needle in haystack(NIH)** (G, 2023). NIH tests the long context capability not only under different context lengths but also at different positions where the correct answer is located in the context" (Appendix B).

## 4. Domain and Modality Scope

- Single domain: Yes, evaluation is framed around language modeling / long-context text tasks ("large language models" and "language modeling") (Section 1 Introduction; Section 4).
- Multiple domains within the same modality: Not specified.
- Multiple modalities: Not claimed.
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Perplexity (PG19) | Not specified. | Not specified. | Not specified. | "For evaluation, we test the long context capabilities comprehensively, the benchmarks are listed below: **perplexity** on PG19 (Rae et al., 2019) test split." (Appendix B) |
| Long-eval | Not specified. | Not specified. | Not specified. | "**Long-eval** (Li\* et al., 2023). This test generates massive random similar sentences and asks the model to answer questions according to a specific sentence in the context." (Appendix B) |
| Retrieval | Not specified. | Not specified. | Not specified. | "Retrieval (Mohtashami & Jaggi, 2024)" (Appendix B) |
| Needle in Haystack (NIH) | Not specified. | Not specified. | Not specified. | "**needle in haystack(NIH)** (G, 2023). NIH tests the long context capability not only under different context lengths but also at different positions where the correct answer is located in the context" (Appendix B) |

## 6. Input and Representation Constraints

- Sequence length is set explicitly in training/fine-tuning (fixed lengths stated): "Training length 32K" and "Training length 4K" (Table 5); "we fine-tune Llama2-7B with a small base (500) to a context length of 32k" (Section 3, Figure 3 caption).
- Context-length ranges are explicitly enumerated in theory tables: "Context Len. | 1k ... 1M" (Table 2).
- Fixed patch size: Not specified.
- Fixed number of tokens per input beyond stated training/context lengths: Not specified.
- Fixed dimensionality (e.g., strictly 2D): Not specified.
- Padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length mentioned: "Context Len. ... 1M" (Table 2); "low perplexity even at 128k context length" (Section 3).
- Fixed or variable sequence length: Fixed lengths are specified for training/fine-tuning ("Training length 32K" and "Training length 4K") with no explicit statement about variable-length handling (Table 5).
- Attention type: Not explicitly categorized; described with standard Transformer attention equations ("The core component of it is the calculation of the attention mechanism" and "A_{ij} = q_i^T k_j") (Section 2.1 Attention and RoPE).
- Mechanisms to manage computational cost: "Both training and testing are accelerated by FlashAttention-2 (Dao et al., 2022) and Megatron-LM (Shoeybi et al., 2020)." (Appendix B).

## 8. Positional Encoding (Critical Section)

- Mechanism: RoPE (rotary position embedding). Quote: "Rotary position embedding (RoPE), a technique that encodes the position information with a rotation matrix" (Abstract).
- Where applied: RoPE is applied in attention score computation via rotation of queries/keys: "RoPE ... applies rotation matrix into the calculation of the attention score in Eq. 1" with "A_{ij} = (R_{i,\theta}q_i)^T (R_{j,\theta}k_i)" (Section 2.1 Attention and RoPE).
- Fixed vs modified across experiments: RoPE base is varied across experiments (e.g., "we fine-tune Llama2-7B with a small base (500) to a context length of 32k" (Section 3); "Llama2-7B-Base with base=1e4" and "Llama2-7B-Base with base=2e5" (Figure 10 and Figure 11 captions)).

## 9. Positional Encoding as a Variable

- Core research variable: Yes, the paper centers on RoPE base as a variable ("we derive that the base of RoPE bounds context length: there is an absolute lower bound for the base value to obtain certain context length capability") (Abstract).
- Multiple positional encodings compared: Multiple RoPE base values are compared ("small base (500)" (Section 3); "base=1e4" and "base=2e5" (Figure 10 and Figure 11 captions)); no explicit comparison to non-RoPE encodings is stated.
- PE choice claimed non-critical: Not claimed.

## 10. Evidence of Constraint Masking

- Model sizes used: "Llama2-7B ... Baichuan2-7B ... a 2-billion model we trained from scratch" (Section 1 Introduction); "Llama2-7B" and "Baichuan2-7B" and "a 2B model from scratch" (Appendix B).
- Dataset sizes: "Training tokens | 4B ... 1T" (Table 5) and "The dataset of both fine-tuning and training from scratch is a subset of RedPajama" (Appendix B).
- Performance attribution focuses on RoPE base rather than scaling: "the base of RoPE bounds context length" (Abstract); "the model may show superficial long context capability ... can only preserve low perplexity but loses the ability to retrieve long context information" (Section 3 Motivation).

## 11. Architectural Workarounds

- Positional-embedding adjustments for long context are discussed: "PI" and "NTK-series" are described as methods that "extend the long context ability of LLMs" by modifying position embedding/base (Section 2.2 OOD theory of relative rotation angle).
- Engineering acceleration used in experiments: "Both training and testing are accelerated by FlashAttention-2 ... and Megatron-LM" (Appendix B).
- No windowed/hierarchical attention, token pooling, or task-specific heads are explicitly introduced in the paper's method (not stated).

## 12. Explicit Limitations and Non-Claims

- Limitations: "the existence of the upper bound for RoPE's base remains an open question that warrants further exploration" and "because of the lack of effective benchmarks for assessing long-context capabilities, the scope of long-context capabilities discussed in this paper may be limited" (Section 7 Limitation).
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Single-domain language modeling evaluations on long-context text benchmarks (Appendix B; Section 4).
> - Task structure: Multiple long-context evaluation tasks (perplexity, Long-eval, Retrieval, NIH) rather than unconstrained multi-domain tasks (Appendix B).
> - Representation rigidity: Fixed training/context lengths are specified ("Training length 32K"; "context length of 32k") (Table 5; Section 3).
> - Model sharing vs specialization: Same LLMs are evaluated across benchmarks with no task-specific heads specified (Appendix B).
> - Role of positional encoding: Central variable with RoPE base bounds on context length ("base of RoPE bounds context length") (Abstract).

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple long-context benchmarks ("perplexity on PG19," "Long-eval," "Retrieval," and "needle in haystack(NIH)") within language modeling (Appendix B; Section 4). It does not claim multiple modalities or cross-domain transfer (Not claimed), so the scope remains single-domain despite multiple tasks.
