## 1. Basic Metadata

- Title: "TransXSSM: A Hybrid Transformer–State Space Model with Unified Rotary Position Embedding" (Title)
- Authors: "$\\begin{array}{ccc} \\textbf{Bingheng Wu}^1 & \\textbf{Jingze Shi}^1 & \\textbf{Yifan Wu}^1 \\\\ & \\textbf{Nan Tang}^1 & \\textbf{Yuyu Luo}^1 \\\\ & {}^1 \\textbf{HKUST (Guangzhou)} \\end{array}$" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper's primary contribution is to "propose a unified rotary position embedding (Unified RoPE) methodology, thereby establishing a consistent positional encoding framework for both self-attention and state-space components" and, "Using this Unified RoPE," to "introduce TransXSSM, a hybrid architecture that coherently integrates the Transformer and SSM layers under this unified positional encoding scheme" (Abstract).

---

## 3. Tasks Evaluated

- Task name: Language modeling benchmarks. Task type: Generation. Dataset(s) used: "Smollm-Corpus [21] dataset" (B.1 Model Architectures Settings); evaluation dataset not specified. Domain: language/text ("sequence modeling in language tasks" (Introduction)). Quote: "it surpasses a Transformer baseline by over 4% on language modeling benchmarks." (Abstract); "For language modeling, TransXSSM follows the standard Transformer architecture outline with modifications to include SSM layers." (Section 3)
- Task name: Needle-in-a-haystack long-context retrieval. Task type: Other (retrieval). Dataset(s) used: synthetic "needle in a haystack" task. Domain: language/text ("random sentence" in a "long document"). Quote: "Long-Context Retrieval (Needle-in-a-Haystack Task). We further evaluated architectures on the \"needle in a haystack\" synthetic retrieval task. This task tests long-context extraction by embedding a \"needle\" (random sentence) in a \"haystack\" (long document) for retrieval." (Section 4.2)
- Task name: MMLU. Task type: Other (task type not specified in paper). Dataset(s) used: "MMLU [25]" (B.1 Downstream Tasks Settings). Domain: language/text ("sequence modeling in language tasks" (Introduction)). Quote: "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings)
- Task name: TriviaQA. Task type: Other (task type not specified in paper). Dataset(s) used: "TriviaQA [26]" (B.1 Downstream Tasks Settings). Domain: language/text ("sequence modeling in language tasks" (Introduction)). Quote: "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings)
- Task name: ARC. Task type: Other (task type not specified in paper). Dataset(s) used: "ARC [27]" (B.1 Downstream Tasks Settings). Domain: language/text ("sequence modeling in language tasks" (Introduction)). Quote: "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings)
- Task name: PIQA. Task type: Reasoning / relational. Dataset(s) used: "PIQA [28]" (B.1 Downstream Tasks Settings). Domain: language/text ("sequence modeling in language tasks" (Introduction)). Quote: "Enhanced Reasoning Capabilities: TransXSSM leads on tasks requiring commonsense reasoning and contextual understanding (e.g., HellaSwag, PIQA, Winogrande)." (Section 4.2); "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings)
- Task name: HellaSwag. Task type: Reasoning / relational. Dataset(s) used: "HellaSwag [29]" (B.1 Downstream Tasks Settings). Domain: language/text ("sequence modeling in language tasks" (Introduction)). Quote: "Enhanced Reasoning Capabilities: TransXSSM leads on tasks requiring commonsense reasoning and contextual understanding (e.g., HellaSwag, PIQA, Winogrande)." (Section 4.2); "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings)
- Task name: OBQA. Task type: Other (task type not specified in paper). Dataset(s) used: "OBQA [30]" (B.1 Downstream Tasks Settings). Domain: language/text ("sequence modeling in language tasks" (Introduction)). Quote: "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings)
- Task name: Winogrande. Task type: Reasoning / relational. Dataset(s) used: "Winogrande [31]" (B.1 Downstream Tasks Settings). Domain: language/text ("sequence modeling in language tasks" (Introduction)). Quote: "Enhanced Reasoning Capabilities: TransXSSM leads on tasks requiring commonsense reasoning and contextual understanding (e.g., HellaSwag, PIQA, Winogrande)." (Section 4.2); "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings)

---

## 4. Domain and Modality Scope

Single domain? The paper frames the work as "sequence modeling in language tasks" (Introduction) and evaluates only language benchmarks ("The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings)), so the evaluated domain is language/text.

Multiple domains within the same modality? Yes, multiple language tasks are evaluated within the same text modality ("The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings)).

Multiple modalities? Not indicated; only language tasks are mentioned (Introduction; B.1 Downstream Tasks Settings).

Domain generalization or cross-domain transfer? Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Language modeling benchmarks | Not specified. | Not specified. | Not specified. | "All models were trained from scratch on the Smollm-Corpus [21] dataset using the NeoX tokenizer [22]." (B.1 Model Architectures Settings); "it surpasses a Transformer baseline by over 4% on language modeling benchmarks." (Abstract) |
| Needle-in-a-haystack long-context retrieval | Not specified. | Not specified. | Not specified. | "We further evaluated architectures on the \"needle in a haystack\" synthetic retrieval task." (Section 4.2) |
| MMLU | Not specified. | Not specified. | Not specified. | "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings) |
| TriviaQA | Not specified. | Not specified. | Not specified. | "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings) |
| ARC | Not specified. | Not specified. | Not specified. | "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings) |
| PIQA | Not specified. | Not specified. | Not specified. | "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings) |
| HellaSwag | Not specified. | Not specified. | Not specified. | "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings) |
| OBQA | Not specified. | Not specified. | Not specified. | "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings) |
| Winogrande | Not specified. | Not specified. | Not specified. | "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings) |

---

## 6. Input and Representation Constraints

- Fixed/variable input resolution: Not specified (no image inputs described; the paper focuses on "language tasks" (Introduction)).
- Fixed patch size: Not specified.
- Fixed number of tokens / sequence length: "At a 4K sequence length" (Abstract); "sequence length 8192" (Section 4.1); "training TransXSSM on contexts up to 16K tokens" (Section 3).
- Fixed dimensionality: "For the 320M scale, models have  $d_{model}=768" and "For the 1.3B scale,  $d_{model}=2048" (B.1 Model Scales Settings); theoretical derivation assumes "the word embedding vector dimension is d=2" (Appendix A).
- Chunking constraint for SSMs: "State-Space components within Mamba2, Jamba, and TransXSSM uniformly use  $d_{state}=128$  and  $chunk\_len=256$." (B.1 Model Scales Settings)
- Padding or resizing requirements: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length: "training TransXSSM on contexts up to 16K tokens" (Section 3).
- Fixed or variable sequence length: Fixed lengths are reported for experiments ("At a 4K sequence length" (Abstract); "sequence length 8192" (Section 4.1)); variable-length handling is not stated.
- Attention type: Causal global self-attention with masking: "a binary lower-triangular mask L is applied to ensure each position only attends to previous positions" (Section 2.1).
- Mechanisms to manage computational cost: SSM layers for linear-time/near-linear scaling and a hybrid stacking ratio: "State Space Models (SSMs) facilitate linear-time sequence modeling" (Abstract); "the SSM layers efficiently handle the bulk of sequence length (contributing near-linear time complexity and high throughput), while the periodic attention layers inject global context mixing" (Section 3, Principle 2); "each module consists of 7 SSM-based sub-layers followed by 1 Transformer attention sub-layer" (Section 3).

---

## 8. Positional Encoding (Critical Section)

- Mechanism: Unified Rotary Position Embedding (RoPE): "We propose a unified rotary position encoding that applies the same rotational embedding to both self-attention (Transformer) and state-space (SSM) components." (Section 2.2); "Rotary Position Embedding (RoPE) is a technique that encodes absolute positions as complex rotations applied to query and key vectors." (Section 2.1)
- Where applied: Applied to both attention and SSM updates and used in every layer: "By applying the same rotary positional transformations to the state update signals of an SSM as we do to the query/key vectors of self-attention, we establish a single consistent positional encoding that is shared across both module types." (Section 2.2); "Every SS and SA sub-layer in TransXSSM uses the same Unified RoPE as described in Section 2.2." (Section 3, Principle 1)
- Fixed vs ablated/compared: Compared against alternatives in ablations: "We evaluated Self-Attention, State-Space, and a hybrid State-Space + Self-Attention setup using three schemes: Conv1d + D (1D convolution + dense layer, State-Space only),  $a_t$  (recursive, State-Space only), and our proposed  $Unified\ RoPE$ ." (Section 4.1)

---

## 9. Positional Encoding as a Variable

- Core research variable? Yes: "(RQ1) Does Unified RoPE effectively unify position encoding across Transformer and SSM modules, and how does it compare with other positional encodings?" (Section 4)
- Multiple positional encodings compared? Yes: "using three schemes: Conv1d + D ...  $a_t$  ... and our proposed  $Unified\ RoPE$ ." (Section 4.1)
- PE choice claimed as not critical or secondary? Not stated.

---

## 10. Evidence of Constraint Masking

- Model size(s): "two model scales, 320M and 1.3B parameters" (B.1 Model Scales Settings); "TransXSSM-1.3B gains 7.22% in average accuracy over its 320M version" (Abstract).
- Dataset size(s): Not specified; only the dataset name is given: "Smollm-Corpus [21] dataset" (B.1 Model Architectures Settings).
- Performance gains attributed to scaling model size: "TransXSSM-1.3B gains 7.22% in average accuracy over its 320M version" (Abstract); "advantages grow with model scale" (Section 4.3).
- Performance gains attributed to architecture: "Our results show that unified positional encoding resolves positional incompatibility in hybrid models, enabling efficient, high-performance long-context modeling." (Abstract); "TransXSSM's RoPE-facilitated consistent position representation across components is crucial for coherent contextual understanding and superior performance" (Section 4.2).
- Training tricks or stabilizers: "We employ RMSNorm normalization and residual skip connections around every sub-layer (both SS and SA) and its FFN." (Section 3, Principle 4)

---

## 11. Architectural Workarounds

- Hybrid stacking ratio for efficiency vs global reasoning: "each module consists of 7 SSM-based sub-layers followed by 1 Transformer attention sub-layer" (Section 3); purpose stated as "SSM layers efficiently handle the bulk of sequence length ... while the periodic attention layers inject global context mixing and strong relational reasoning" (Section 3, Principle 2).
- Unified RoPE across modules to avoid positional discontinuity: "All modules share the same Unified RoPE encoding" (Section 3).
- Causal masking for autoregressive control: "a binary lower-triangular mask L is applied to ensure each position only attends to previous positions" (Section 2.1).
- Stabilization for long contexts: "We employ RMSNorm normalization and residual skip connections around every sub-layer (both SS and SA) and its FFN" and this is "crucial for training TransXSSM on contexts up to 16K tokens without divergence." (Section 3, Principle 4)
- SSM chunking constraint: "State-Space components within Mamba2, Jamba, and TransXSSM uniformly use  $d_{state}=128$  and  $chunk\_len=256$." (B.1 Model Scales Settings)

---

## 12. Explicit Limitations and Non-Claims

No explicit limitations, future work, or non-claims are stated in the provided text.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Language/text only, framed as "sequence modeling in language tasks" (Introduction) and evaluated on language benchmarks ("The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings)).
> – Task structure: Multiple downstream benchmarks plus a synthetic retrieval task ("Downstream task evaluation utilized the EleutherAI LM Evaluation Harness [24]. The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings); "needle in a haystack" task description (Section 4.2)).
> – Representation rigidity: Fixed sequence lengths are reported ("At a 4K sequence length" (Abstract); "sequence length 8192" (Section 4.1)) and training reaches "contexts up to 16K tokens" with fixed "chunk\_len=256" (Section 3; B.1 Model Scales Settings).
> – Model sharing vs specialization: The paper reports evaluation across benchmarks but does not specify per-task fine-tuning or separate heads ("Downstream task evaluation utilized the EleutherAI LM Evaluation Harness [24]." (B.1 Downstream Tasks Settings)).
> – Role of positional encoding: Central and unified across layers ("Every SS and SA sub-layer in TransXSSM uses the same Unified RoPE" (Section 3, Principle 1); "(RQ1) Does Unified RoPE effectively unify position encoding across Transformer and SSM modules, and how does it compare with other positional encodings?" (Section 4)).

---

## 14. Final Classification

**Multi-task, single-domain.** The evaluation is confined to language/text tasks, explicitly framed as "sequence modeling in language tasks" (Introduction) and tested on a suite of language benchmarks ("The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]." (B.1 Downstream Tasks Settings)). It also includes a synthetic long-context retrieval task ("needle in a haystack" task) (Section 4.2), indicating multiple tasks within a single modality rather than multi-domain or multi-modality evaluation.
