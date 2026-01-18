## 1. Basic Metadata

- Title: Extending Context Window of Large Language Models from a Distributional Perspective
- Authors: Yingsheng Wu; Yuxuan Gu; Xiaocheng Feng; Weihong Zhong; Dongliang Xu; Qing Yang; Hongtao Liu; Bing Qin
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper proposes extending RoPE-based LLM context windows by minimizing disturbance to rotary angle distributions to preserve pre-training consistency.

## 3. Tasks Evaluated

- Task name: Longbench-E benchmark (13 diverse tasks)
  - Task type: Other (long-context benchmark; task type not specified in paper)
  - Dataset(s) used: Longbench-E
  - Domain: Natural language text (long context)
  - Quotes (Section 4.2 Long Context Evaluation): "We utilize the Longbench-E benchmark (Bai et al., 2023), which is specifically designed for evaluating models with long context window. The Longbench-E benchmark consists of 13 diverse tasks, with the average length of most tasks ranging from 5k to 15k."

- Task name: TruthfulQA (0-shot)
  - Task type: Other (task type not specified in paper)
  - Dataset(s) used: TruthfulQA
  - Domain: Natural language text
  - Quotes (Section 4.3 Short Context Validation): "Specifically, we use 0-shot TruthfulQA (Lin et al., 2022) and Hellaswag (Zellers et al., 2019), 5-shot MMLU (Hendrycks et al., 2020) and 25-shot ARC-c (Clark et al., 2018)."

- Task name: Hellaswag (0-shot)
  - Task type: Other (task type not specified in paper)
  - Dataset(s) used: Hellaswag
  - Domain: Natural language text
  - Quotes (Section 4.3 Short Context Validation): "Specifically, we use 0-shot TruthfulQA (Lin et al., 2022) and Hellaswag (Zellers et al., 2019), 5-shot MMLU (Hendrycks et al., 2020) and 25-shot ARC-c (Clark et al., 2018)."

- Task name: MMLU (5-shot)
  - Task type: Other (task type not specified in paper)
  - Dataset(s) used: MMLU
  - Domain: Natural language text
  - Quotes (Section 4.3 Short Context Validation): "Specifically, we use 0-shot TruthfulQA (Lin et al., 2022) and Hellaswag (Zellers et al., 2019), 5-shot MMLU (Hendrycks et al., 2020) and 25-shot ARC-c (Clark et al., 2018)."

- Task name: ARC-c (25-shot)
  - Task type: Other (task type not specified in paper)
  - Dataset(s) used: ARC-c
  - Domain: Natural language text
  - Quotes (Section 4.3 Short Context Validation): "Specifically, we use 0-shot TruthfulQA (Lin et al., 2022) and Hellaswag (Zellers et al., 2019), 5-shot MMLU (Hendrycks et al., 2020) and 25-shot ARC-c (Clark et al., 2018)."

- Task name: Passkey retrieval
  - Task type: Other (retrieval)
  - Dataset(s) used: Passkey retrieval task (prompt template)
  - Domain: Natural language text (synthetic long prompts)
  - Quotes (Section 4.4 Passkey Retrieval): "We further evaluate the model's ability to retrieve a simple passkey from a massive amount of text via passkey retrieval task (Mohtashami and Jaggi, 2023)."
  - Quotes (Section B.3 Passkey Prompt): "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."

- Task name: RULER benchmark (long-context retrieval)
  - Task type: Other (retrieval)
  - Dataset(s) used: RULER
  - Domain: Natural language text (long documents)
  - Quotes (Section B.2.1 RULER Benchmark): "The RULER (Hsieh et al., 2024) benchmark is employed to evaluate the long-context retrieval capabilities of models, with the performance of different methods on this benchmark presented in Table 8."

- Task name: Perplexity evaluation (language modeling)
  - Task type: Other (language modeling perplexity)
  - Dataset(s) used: PG19
  - Domain: Natural language text
  - Quotes (Section B.2.3 Perplexity): "Perplexity is commonly employed to evaluate a model's language modeling capabilities, and we tested the perplexity of different methods under non-training conditions, with the results presented in Table 10."
  - Quotes (Section B.2.3 Perplexity): "Table 10: Sliding window perplexity (S = 256) on PG19 dataset."

## 4. Domain and Modality Scope

- Evaluation performed on: Single domain (natural language text) across multiple tasks/benchmarks.
  - Evidence (Introduction): "modeling arbitrarily long textual sequences remains a significant challenge."
  - Evidence (Section 4.2 Long Context Evaluation): "long context tasks"
  - Evidence (Section 4.3 Short Context Validation): "standard short context benchmark"
- Multiple domains within the same modality? Not stated; all evaluations are in text.
- Multiple modalities? Not stated.
- Domain generalization or cross-domain transfer claims? Not claimed; the paper instead notes length generalization (Abstract): "enhancing the model's capability to generalize to longer sequences."

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Longbench-E benchmark (13 diverse tasks) | Not specified. | Yes (context window extension fine-tuning). | Not specified. | Section 4.1 Experimental Details: "All models are trained on a subset of PG19 (Rae et al., 2020) datasets. For s=2, models are fine-tuned for 1000 steps with a global batch size of 64 and max length of 8192."; Section 4.2 Long Context Evaluation: "We utilize the Longbench-E benchmark (Bai et al., 2023)." |
| TruthfulQA (0-shot) | Not specified. | Yes (context window extension fine-tuning). | Not specified. | Section 4.1 Experimental Details: "All models are trained on a subset of PG19 (Rae et al., 2020) datasets."; Section 4.3 Short Context Validation: "Specifically, we use 0-shot TruthfulQA (Lin et al., 2022) and Hellaswag (Zellers et al., 2019), 5-shot MMLU (Hendrycks et al., 2020) and 25-shot ARC-c (Clark et al., 2018)." |
| Hellaswag (0-shot) | Not specified. | Yes (context window extension fine-tuning). | Not specified. | Section 4.1 Experimental Details: "All models are trained on a subset of PG19 (Rae et al., 2020) datasets."; Section 4.3 Short Context Validation: "Specifically, we use 0-shot TruthfulQA (Lin et al., 2022) and Hellaswag (Zellers et al., 2019), 5-shot MMLU (Hendrycks et al., 2020) and 25-shot ARC-c (Clark et al., 2018)." |
| MMLU (5-shot) | Not specified. | Yes (context window extension fine-tuning). | Not specified. | Section 4.1 Experimental Details: "All models are trained on a subset of PG19 (Rae et al., 2020) datasets."; Section 4.3 Short Context Validation: "Specifically, we use 0-shot TruthfulQA (Lin et al., 2022) and Hellaswag (Zellers et al., 2019), 5-shot MMLU (Hendrycks et al., 2020) and 25-shot ARC-c (Clark et al., 2018)." |
| ARC-c (25-shot) | Not specified. | Yes (context window extension fine-tuning). | Not specified. | Section 4.1 Experimental Details: "All models are trained on a subset of PG19 (Rae et al., 2020) datasets."; Section 4.3 Short Context Validation: "Specifically, we use 0-shot TruthfulQA (Lin et al., 2022) and Hellaswag (Zellers et al., 2019), 5-shot MMLU (Hendrycks et al., 2020) and 25-shot ARC-c (Clark et al., 2018)." |
| Passkey retrieval | Not specified. | Yes (context window extension fine-tuning). | Not specified. | Section 4.1 Experimental Details: "All models are trained on a subset of PG19 (Rae et al., 2020) datasets."; Section 4.4 Passkey Retrieval: "We further evaluate the model's ability to retrieve a simple passkey from a massive amount of text via passkey retrieval task (Mohtashami and Jaggi, 2023)." |
| RULER benchmark (long-context retrieval) | Not specified. | Yes (context window extension fine-tuning). | Not specified. | Section 4.1 Experimental Details: "All models are trained on a subset of PG19 (Rae et al., 2020) datasets."; Section B.2.1 RULER Benchmark: "The RULER (Hsieh et al., 2024) benchmark is employed to evaluate the long-context retrieval capabilities of models" |
| Perplexity evaluation (language modeling) | Not specified. | Yes (context window extension fine-tuning). | Not specified. | Section 4.1 Experimental Details: "All models are trained on a subset of PG19 (Rae et al., 2020) datasets."; Section B.2.3 Perplexity: "Perplexity is commonly employed to evaluate a model's language modeling capabilities" |

## 6. Input and Representation Constraints

- Sequence length and token representation are explicit: Section 2.1 Rotary Position Embedding (RoPE): "Suppose the input of a single attention head is  $x_1, \dots, x_l \in \mathbb{R}^d$ , where l is the sequence length and d is the dimension of an attention head."
- Pretraining length and scaling factor are defined: Section 2.2 Position Interpolation (PI): "When extending the context window from L to L', with the scaling factor s = L'/L, the new  $\hat{\theta}_i$  is scaled correspondingly as  $\hat{\theta}_i = \theta_i/s$ ."
- Maximum lengths for fine-tuning are specified: Section 4.1 Experimental Details: "For s=2, models are fine-tuned for 1000 steps with a global batch size of 64 and max length of 8192. For s=4, models are fine-tuned for 500 steps with a global batch size of 64 and a max length of 16384."
- Maximum input length for passkey retrieval is specified: Section 4.4 Passkey Retrieval: "we set the maximum input length for all models to 20k"
- Padding, resizing, fixed patch size, or fixed token count beyond max length: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length(s) used in experiments: 8k/16k fine-tuning and 20k evaluation. Evidence (Section 4.1 Experimental Details): "max length of 8192" and "a max length of 16384." Evidence (Section 4.4 Passkey Retrieval): "we set the maximum input length for all models to 20k"
- Fixed or variable sequence length: variable, with explicit length buckets and varying prompt length. Evidence (Section 4.2 Long Context Evaluation): "categorizes the test samples into groups based on length intervals of 0-4k, 4-8k, and 8k+" Evidence (Section B.3 Passkey Prompt): "the prompt length varies with the value of n and m."
- Attention type: Not explicitly stated; RoPE is applied in standard attention computation. Evidence (Section 2.1 Rotary Position Embedding (RoPE)): "the attention logit  $\mathbf{q}_m^{\top}\mathbf{k}_n$  with RoPE can be calculate as follows:"
- Computational cost management: No new structural efficiency mechanism; they note compatibility and acceleration. Evidence (Section 8 Limitations): "the quadratic computational complexity problem of transformers still exists. Fortunately, our method does not introduce more computational overhead in the inference phase." Evidence (Section B.1 Experimental Setup): "Both training and testing are accelerated by FlashAttention-2 (Dao, 2023)."

## 8. Positional Encoding (Critical Section)

- Mechanism used: RoPE. Evidence (Section 2.1 Rotary Position Embedding (RoPE)): "Rotary position embedding (Su et al., 2021) is a position embedding method widely used in recent LLMs"
- Where applied: In attention via rotation of queries/keys. Evidence (Section 2.1 Rotary Position Embedding (RoPE)): "$$\mathbf{q}_{m}^{\top}\mathbf{k}_{n} = (\mathcal{R}_{m}^{d}\mathbf{W}_{q}x_{m})^{\top}(\mathcal{R}_{n}^{d}\mathbf{W}_{k}x_{n})$$"
- Fixed vs modified: RoPE angles are modified for context extension; multiple scaling strategies are defined. Evidence (Section 2.2 Position Interpolation (PI)): "the new  $\hat{\theta}_i$  is scaled correspondingly as  $\hat{\theta}_i = \theta_i/s$ ." Evidence (Section 3.2 Minimizing Distribution Disturbance): "Thus, we modify the rotary position embedding as follows:" followed by the piecewise definition of $\hat{\theta}_i$.
- Ablated or compared against alternatives: Yes, compared to other RoPE scaling methods. Evidence (Table 1 caption): "Comparative performance analysis of various context window extension methods on the Longbench-E benchmark."

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption? Core research variable; the paper studies RoPE scaling. Evidence (Abstract): "Scaling the rotary position embedding (RoPE) has become a common method for extending the context window of RoPE-based large language models (LLMs)." Evidence (Abstract): "we propose to optimize the context window extending task from the view of rotary angle distribution."
- Multiple positional encodings compared? Multiple RoPE scaling methods are compared (PI, YaRN, CLEX, and the proposed method). Evidence (Table 1 caption): "Comparative performance analysis of various context window extension methods on the Longbench-E benchmark."
- Claim that PE choice is not critical or secondary? Not claimed.

## 10. Evidence of Constraint Masking

- Model sizes used: Section 4.1 Experimental Details: "including 7B and 13B parameter models."
- Dataset size(s): Not specified; only "All models are trained on a subset of PG19 (Rae et al., 2020) datasets." (Section 4.1 Experimental Details).
- Performance gains attributed to distributional consistency rather than scale: Abstract: "we present a novel extension strategy that minimizes the disturbance between rotary angle distributions to maintain consistency with the pre-training phase" and "On the LongBench-E benchmark, our method achieves an average improvement of up to 4.33% over existing state-of-the-art methods." Section 5.1 Influence of Disturbance: "with the disturbance increases, the performance of the model basically shows a monotonically decreasing trend."
- Attribution to scaling model size or data volume is not stated.

## 11. Architectural Workarounds

- RoPE scaling via interpolation/extrapolation for context extension: Section 2.2 Position Interpolation (PI): "When extending the context window from L to L', with the scaling factor s = L'/L, the new  $\hat{\theta}_i$  is scaled correspondingly as  $\hat{\theta}_i = \theta_i/s$ ."
- Per-dimension strategy selection to minimize disturbance: Section 3.2 Minimizing Distribution Disturbance: "we combine the two strategies: one is based on PI... the other involves directly extrapolating to L'... We minimize the disturbance score for each dimension independently"
- Efficiency compatibility rather than new architecture: Section B.1 Experimental Setup: "Both training and testing are accelerated by FlashAttention-2 (Dao, 2023)." Section 8 Limitations: "our method does not introduce more computational overhead in the inference phase."

## 12. Explicit Limitations and Non-Claims

- Limitation to RoPE-based models: Section 8 Limitations: "Our method is limited by the rotary position embedding, which is not currently available for LLMs with other embedding methods."
- Transformer quadratic cost remains: Section 8 Limitations: "the quadratic computational complexity problem of transformers still exists."
- No structural improvement to RoPE/interpolation: Section 8 Limitations: "Our method does not make any structural improvements to the rotation position embedding or interpolation methods"
- Distributional estimation depends on pre-training length: Section 8 Limitations: "The accuracy of our estimated rotary angle distribution is affected by the pre-training sequence length L"
- Experimental scale limits: Section 8 Limitations: "our experiments are limited to LLaMA2-7B and LLaMA2-13B, and the long contextual ability is also constrained by the model size."

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Single modality/domain (natural language text) evaluated across multiple benchmarks.
> - Task structure: Multi-task evaluation (Longbench-E with 13 tasks plus short-context benchmarks, retrieval, and perplexity) but all within LLM text tasks.
> - Representation rigidity: Fixed token sequence modeling with explicit max lengths (4k/8k/16k/20k) and fixed attention head dimensionality d; RoPE angle scaling is the main representation change.
> - Model sharing vs specialization: Same LLaMA2 7B/13B models are fine-tuned for context extension and reused across evaluations; no task-specific heads described.
> - Role of positional encoding: Central experimental variable; the method modifies RoPE angles to extend context.

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple tasks/benchmarks within text, including a benchmark that "consists of 13 diverse tasks" (Section 4.2 Long Context Evaluation) and additional short-context benchmarks and retrieval tasks (Sections 4.3 and 4.4). All evaluations are within natural language text (e.g., "textual sequences" in the Introduction), and there are no claims of cross-domain or multi-modal transfer.
