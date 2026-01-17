## 1. Basic Metadata

- Title: "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens" (title header)
- Authors: "Yiran Ding * Li Lyna Zhang † Chengruidong Zhang Yuanyuan Xu * Ning Shang Jiahang Xu Fan Yang Mao Yang Microsoft Research" (front matter)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

"This paper introduces LongRoPE that, for the first time, extends the context window of pre-trained LLMs to an impressive 2048k tokens, with up to only 1k fine-tuning steps at within 256k training lengths, while maintaining performance at the original short context window." (Abstract)

## 3. Tasks Evaluated

### Task 1: Long sequence language modeling / perplexity on long documents
- Task type: Generation
- Dataset(s) used: PG19; Proof-pile; Books3
- Domain: Natural language text (long documents)
- Quotes:
  - "We apply LongRoPE on LLaMA2-7B and Mistral-7B, and evaluate the performance on three aspects: (1) perplexity of extended-context LLMs on long documents" (Section 4.1. Setup)
  - "We use two datasets to demonstrate our generalizability: Proof-pile (Rae et al., 2019) and PG19 (Gao et al., 2020) test splits." (Section 4.2. Main Results)
  - "To evaluate the effectiveness on extremely long documents, we use the Books3 (Gao et al., 2020) dataset." (Section 4.2. Main Results)

### Task 2: Passkey retrieval
- Task type: Generation; Reasoning / relational
- Dataset(s) used: Synthetic passkey retrieval prompt (no named dataset)
- Domain: Synthetic text document with embedded passkey
- Quotes:
  - "Passkey retrieval task that measures a model's ability to retrieve a simple passkey from a sea of irrelevant text" (Section 4.1. Setup)
  - "We follow a synthetic evaluation task of passkey retrieval proposed by (Mohtashami & Jaggi, 2023)." (Section 4.2. Main Results)
  - "The pass key is 17865. Remember it. 17865 is the pass key." (Appendix A.1. Settings)

### Task 3: Standard LLM benchmarks within original context window
- Task type: Classification; Reasoning / relational
- Dataset(s) used: ARC-Challenge; HellaSwag; MMLU; TruthfulQA (via Hugging Face Open LLM Leader-board)
- Domain: Natural language text benchmarks
- Quotes:
  - "Standard LLM benchmarks within a short 4096 context window size." (Section 4.1. Setup)
  - "We evaluate LongRoPE-2048k models on the original context window using Hugging Face Open LLM Leader-board (Face, 2024) in zero-shot and few-shot settings. We use 25-shot ARC-Challenge (Clark et al., 2018). 10-shot HellaSwag (Zellers et al., 2019), 5-shot MMLU (Hendrycks et al., 2020), and 0-shot TruthfulQA (Lin et al., 2021)." (Section 4.2. Main Results)

## 4. Domain and Modality Scope

- Modality: Text-only evaluation, as tasks are "perplexity of extended-context LLMs on long documents," "Passkey retrieval," and "Standard LLM benchmarks" (Section 4.1. Setup).
- Domain scope: Multiple text datasets are used (long documents and benchmark QA), e.g., "Proof-pile" and "PG19" plus "Books3" (Section 4.2. Main Results); this indicates multiple datasets within the same text modality.
- Domain generalization or cross-domain transfer: The paper claims dataset-level generalizability: "We use two datasets to demonstrate our generalizability" (Section 4.2. Main Results). Cross-domain transfer is not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Long sequence language modeling / perplexity | Yes | Yes (context-extension fine-tuning) | Not mentioned | "We apply LongRoPE on LLaMA2-7B and Mistral-7B, and evaluate the performance on three aspects" (Section 4.1. Setup); "We fine-tune for 400 steps on Redpajama (Computer, 2023) dataset" (Section 4.1. Setup) |
| Passkey retrieval | Yes | Yes (context-extension fine-tuning) | Not mentioned | "We apply LongRoPE on LLaMA2-7B and Mistral-7B, and evaluate the performance on three aspects" (Section 4.1. Setup); "We fine-tune for 400 steps on Redpajama (Computer, 2023) dataset" (Section 4.1. Setup) |
| Standard LLM benchmarks | Yes | Yes (context-extension fine-tuning) | Not mentioned | "We apply LongRoPE on LLaMA2-7B and Mistral-7B, and evaluate the performance on three aspects" (Section 4.1. Setup); "We evaluate LongRoPE-2048k models on the original context window" (Section 4.2. Main Results) |

## 6. Input and Representation Constraints

- Fixed/variable sequence length: The evaluation uses multiple lengths: "We evaluate perplexity at various context lengths" (Section 4.2. Main Results), and inference adapts by length: "During inference, if the sequence length is less than 8k, we update RoPE with the searched rescale factors." (Section 3.3)
- Fixed number of tokens during training chunks: "We fine-tune for 400 steps on Redpajama (Computer, 2023) dataset, chunked into 128k segments bookended with the BOS and EOS tokens." (Section 4.1. Setup)
- Training length constraints for Mistral: "we follow the setting in YaRN (Peng et al., 2023), with 400 steps on the Together Computer's Long-Data Collections (mis, 2024) using 16k sequence length." (Section 4.1. Setup)
- Sliding-window evaluation constraints: "We evaluate perplexity at various context lengths using sliding window of 256." (Section 4.2. Main Results) and "use a sliding window of 256k." (Section 4.2. Main Results)
- Fixed patch size, fixed spatial dimensionality, padding/resizing: Not specified (text tokens only).

## 7. Context Window and Attention Structure

- Maximum sequence length: "extends the context window of pre-trained LLMs to an impressive 2048k tokens" (Abstract).
- Fixed or variable: The paper evaluates multiple lengths and adjusts at inference: "We evaluate perplexity at various context lengths" (Section 4.2. Main Results) and "During inference, if the sequence length is less than 8k, we update RoPE with the searched rescale factors." (Section 3.3).
- Attention type: Not explicitly specified; the paper notes that models "retain the original architecture with minor modifications to the positional embedding" (Abstract).
- Mechanisms to manage computational cost: "We employ Flash Attention-2 (Dao, 2023) to accelerate both training and inference." (Appendix A.1. Settings); "we use a sliding window of 256" (Section 4.2. Main Results); "use a sliding window of 256k" (Section 4.2. Main Results); "we utilize an internal platform, CUBE - an internal version of (Lin et al., 2023), to reduce both the training and inference costs." (Appendix A.1. Settings)

## 8. Positional Encoding (Critical Section)

- Mechanism: RoPE (rotary position embedding): "Our work focuses on the RoPE (Su et al., 2021) position embedding" (Section 2.1. Preliminary).
- Where it is applied: The paper describes RoPE as a "position embedding" (Section 2.1. Preliminary) and discusses "RoPE's rotation angles" (Section 1. Introduction), but does not specify layer placement (input-only vs every layer vs attention bias).
- Fixed vs modified: Positional encoding is modified via interpolation and rescaling, e.g., "we identify and exploit two forms of non-uniformities in positional interpolation" (Abstract) and "It identifies effective rescale factors for RoPE's rotation angles for each RoPE dimension" (Section 1. Introduction).
- Compared against alternatives: "we compare the four models with state-of-the-art context window extension baselines, specifically open-sourced LLMs fine-tuned after positional interpolation using PI, NTK and YaRN." (Section 4.1. Setup)

## 9. Positional Encoding as a Variable

- Core research variable: Yes; "Our work focuses on the RoPE (Su et al., 2021) position embedding" (Section 2.1. Preliminary) and "we identify and exploit two forms of non-uniformities in positional interpolation" (Abstract).
- Multiple positional encodings compared: Yes (different interpolation strategies over RoPE): "PI, NTK and YaRN" are compared as baselines (Section 4.1. Setup).
- Claim that PE choice is not critical/secondary: Not claimed.

## 10. Evidence of Constraint Masking

- Model sizes: "We apply LongRoPE on LLaMA2-7B and Mistral-7B" (Section 4.1. Setup).
- Dataset sizes: Not specified.
- Performance gains attributed to: Positional interpolation and progressive extension, not model or data scaling: "This is achieved by three key innovations: (i) we identify and exploit two forms of non-uniformities in positional interpolation through an efficient search, providing a better initialization for fine-tuning and enabling an 8× extension in non-fine-tuning scenarios; (ii) we introduce a progressive extension strategy that first fine-tunes a 256k length LLM and then conducts a second positional interpolation on the fine-tuned extended LLM to achieve a 2048k context window; (iii) we readjust LongRoPE on 8k length to recover the short context window performance." (Abstract)
- Training steps / tricks: "with up to only 1k fine-tuning steps at within 256k training lengths" (Abstract), and "Specifically, we first fine-tune LLaMA2 for 400 steps using the RoPE rescaled factors for 128k. Then, we replace the RoPE rescaled factors to 256k on the finished checkpoint and conduct an additional 600 steps of fine-tuning." (Section 3.3)

## 11. Architectural Workarounds

- Progressive extension strategy to avoid direct 2048k fine-tuning: "we introduce an efficient, progressive method that achieves the target 2048k with just 1k fine-tuning steps at within 256k training length." (Section 3.3)
- Evolutionary search for RoPE rescale factors: "LongRoPE introduces an evolutionary search algorithm with two optimization techniques to boost search efficiency." (Section 1. Introduction)
- Short-context recovery with dynamic RoPE update: "To mitigate this, we perform an extra evolution search on the extended LLM to adjust RoPE rescale factors for short context lengths (e.g., 4k and 8k)." (Section 3.3) "During inference, the LLM dynamically adjusts the corresponding RoPE rescale factors." (Section 3.3)
- Minimal architectural changes: "Models extended via LongRoPE retain the original architecture with minor modifications to the positional embedding" (Abstract)
- Compute optimizations used: "We employ Flash Attention-2 (Dao, 2023) to accelerate both training and inference." (Appendix A.1. Settings)

## 12. Explicit Limitations and Non-Claims

- Fine-tuning difficulty at large extension ratios: "it's challenging to well fine-tune the LLMs under a large extension ratio" (Section 3.3).
- Resource limits for very long contexts: "As the GPU memory and computation time increase exponentially with the sequence length, it's challenging to serve the fine-tuning and inference with context length beyond 512k." (Appendix A.1. Settings)
- Future work: "Preserving only the initial tokens without interpolation becomes non-useful, and we leave this as future work." (Section 4.3. Ablation Results)
- Search-time constraints: "due to the time required for a single perplexity evaluation at 2048k is about 50 minutes, the search iterations are constrained." (Appendix A.3. Additional details on the search)
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Text-only datasets (long documents and standard NLP benchmarks).
> - Task structure: Multiple text tasks (perplexity, passkey retrieval, and standard benchmarks) rather than open-ended multi-domain evaluation.
> - Representation rigidity: Fixed context window sizes with fixed-length training chunks and sliding-window evaluation.
> - Model sharing vs specialization: Same LLaMA2-7B/Mistral-7B weights evaluated across tasks after context-extension fine-tuning; no task-specific heads.
> - Role of positional encoding: Core research variable; RoPE rescaling/interpolation is central to the method.

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates the same LongRoPE models on three text-centric tasks: "perplexity of extended-context LLMs on long documents," "Passkey retrieval," and "Standard LLM benchmarks" (Section 4.1. Setup). All evaluations are within the text modality and do not introduce non-text domains or modalities, so the setup is multi-task but single-domain.
