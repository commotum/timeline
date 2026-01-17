## 1. Basic Metadata

- Title: "HiRoPE: Length Extrapolation for Code Models Using Hierarchical Position" (Title line)
- Authors: "Kechi Zhang, Ge Li, Huangzhao Zhang, Zhi Jin*" (Author line)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper introduces "Hierarchical Rotary Position Embedding (HiRoPE)" that "enhances the traditional rotary position embedding into a hierarchical format based on the hierarchical structure of source code" to address "the limitation of context length in large language models for code-related tasks." (Abstract)

## 3. Tasks Evaluated

Task name: Long code language modeling
Task type: Generation
Dataset(s) used: CodeParrot-valid (CodeParrot)
Domain: Source code
Evidence: "How is the language modeling capability of HiRoPE on long code sequences? We evaluate HiRoPE's language modeling ability on CodeParrot-valid dataset (CodeParrot, 2022) in Section 5.1." (Section 4 Experiment Setup)

Task name: Long text (natural language) language modeling
Task type: Generation
Dataset(s) used: ReRoPE-eval (Common Crawl)
Domain: Natural language text
Evidence: "How is the language modeling capability of HiRoPE on long natural language sequences?" and "We use the evaluation dataset from ReRoPE-eval (Su, 2023) in Section 5.2. It is a dataset curated from Common Crawl (Crawl, 2023), refined by length-based selection criteria." (Section 4 Experiment Setup)

Task name: Code Symbol Understanding
Task type: Generation; Other (symbol extraction)
Dataset(s) used: Real-world Code Project
Domain: Source code (real-world code projects)
Evidence: "we design a new evaluation task on real code projects: Code Symbol Understanding in Section 5.3. Given a long code file, the model is required to output all the function names and class names defined in it." (Section 4 Experiment Setup) and "Code Symbol Understanding | Real-world Code Project" (Table 1)

Task name: Long code completion
Task type: Generation
Dataset(s) used: LCC; RepoBench
Domain: Source code
Evidence: "We further perform the evaluation using two long code completion benchmarks: LCC (Guo et al., 2023) and RepoBench (Liu et al., 2023a) in Section 5.4." (Section 4 Experiment Setup) and "Given a long code context, the model is required to generate the complete next line of code." (Section 5.4 Long Code Completion)

## 4. Domain and Modality Scope

- Single domain? No. Evidence: "long code sequences" and "long natural language sequences" (Section 4 Experiment Setup).
- Multiple domains within the same modality? Yes; code and natural language text are both evaluated. Evidence: "CodeParrot-valid dataset" and "ReRoPE-eval... curated from Common Crawl" (Section 4 Experiment Setup).
- Multiple modalities? Not claimed.
- Domain generalization or cross-domain transfer? Not claimed; only a general statement that "The results reflect the practicality and generalization ability of our method." (Section 5.4 Long Code Completion)

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Long code language modeling | Yes | No (training-free / no additional training costs) | Not specified | "HiRoPE is a plug-and-play solution, easily integrated into existing LLMs without additional training costs." (Abstract) and "we focus on those popular length extrapolation methods without training" (Section 4.2 Baselines) |
| Long text language modeling | Yes | No (training-free / no additional training costs) | Not specified | "HiRoPE is a plug-and-play solution, easily integrated into existing LLMs without additional training costs." (Abstract) and "we focus on those popular length extrapolation methods without training" (Section 4.2 Baselines) |
| Code Symbol Understanding | Yes | No (training-free / no additional training costs) | Not specified | "HiRoPE is a plug-and-play solution, easily integrated into existing LLMs without additional training costs." (Abstract) and "we focus on those popular length extrapolation methods without training" (Section 4.2 Baselines) |
| Long code completion | Yes | No (training-free / no additional training costs) | Not specified | "HiRoPE is a plug-and-play solution, easily integrated into existing LLMs without additional training costs." (Abstract) and "we focus on those popular length extrapolation methods without training" (Section 4.2 Baselines) |

## 6. Input and Representation Constraints

- Fixed or variable input resolution? Not specified.
- Fixed patch size? Not specified.
- Fixed number of tokens? The paper specifies pretraining context lengths and evaluation length ranges, e.g., "LLMs are typically pre-trained with a context length ranging from 2k to 16k tokens" (Section 1 Introduction) and "0-2048... 2048-4096... 4096-8192... 8192-16384" for long code language modeling (Table 1).
- Fixed dimensionality (e.g., strictly 2D)? Not specified.
- Padding or resizing requirements? Not specified.
- Hierarchical representation constraint: "we use a h-dimensional vector to represent the hierarchical position index from high-level to low-level" (Section 3.1 Hierarchical format) and "Our hierarchical position includes both the token and function/class levels of the source code." (Section 5.5 Ablation Study)
- Natural language segmentation for hierarchy: "we set every 128 tokens as a segment, and encode it as higher-level position information." (Section 4 Experiment Setup)
- Window length constraint in attention computation: "shorter than a specific length L_window" (Section 3.2 Window Mechanism) and "choose a window length L_window = 512." (Section 5.5 Ablation Study)

## 7. Context Window and Attention Structure

- Maximum sequence length: Not explicitly stated; training lengths are listed as "L_pretrain 4096" and "L_pretrain 2048" for base LLMs (Table 2), and evaluation lengths include ranges up to "8192-16384" for long code language modeling (Table 1).
- Fixed or variable sequence length: Variable, with "arbitrary input lengths" noted for HiRoPE usage. (Section 3.2 Window Mechanism)
- Attention type: Not explicitly labeled (global/windowed/sparse); the paper states, "At each layer, RoPE is applied on both query and key embeddings for computing attention scores." (Section 2.1 Rotary Position Embedding in Transformer)
- Mechanisms introduced to manage computational cost: Not stated; the paper introduces a "window mechanism" for stability, not cost: "we also add a window mechanism, so that when dealing with short texts, our proposed method is consistent with the original positional encoding." (Section 3 Hierarchical RoPE)

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: RoPE and hierarchical RoPE. Evidence: "the rotary position embedding (RoPE)" (Section 2.1) and "we introduce Hierarchical Rotary Position Embedding (HiRoPE), a novel approach that enhances the traditional rotary position embedding into a hierarchical format" (Abstract).
- Where it is applied: "At each layer, RoPE is applied on both query and key embeddings for computing attention scores." (Section 2.1 Rotary Position Embedding in Transformer)
- Fixed across experiments or modified: Modified for natural language evaluation and ablated. Evidence: "The natural language lacks the explicit hierarchical structure information found in code, so we have made some modifications: we set every 128 tokens as a segment, and encode it as higher-level position information." (Section 4 Experiment Setup) and "we carry out extensive ablation studies that include the dimensions' split settings, the window mechanism, and the high-level segment split strategy" (Section 4 Experiment Setup).

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption? Core research variable: "We propose Hierarchical RoPE (HiRoPE), enhancing the traditional rotary position embedding into a hierarchical format" (Contributions).
- Multiple positional encodings compared? Yes: "we focus on those popular length extrapolation methods without training, including NTK... ReRoPE... and Self-Extend" and "We also make comparisons with the original RoPE" (Section 4.2 Baselines).
- PE choice claimed as not critical or secondary? Not stated.

## 10. Evidence of Constraint Masking

- Model size(s): "Para. 7B" (LLaMA-2), "1.3B" (ShearedLLaMA), "1.1B" (TinyLLaMA), "7B" (Vicuna). (Table 2)
- Dataset size(s): "Samples 100... 100... 100... 100" for CodeParrot ranges, "200" for ReRoPE-eval, "56" for Code Symbol Understanding, and "300... 300" for LCC/RepoBench. (Table 1)
- Attribution of performance gains: The paper attributes gains to positional/architectural hierarchy, e.g., "Our Hi-RoPE has been improved from the perspective of positional encoding, enabling the model to perceive structural hierarchy changes in the code, thus achieving relatively good results." (Section 5.3 Code Symbol Understanding) and "We propose HiRoPE... enhancing the traditional rotary position embedding into a hierarchical format" (Contributions). It also emphasizes no extra training: "HiRoPE is a plug-and-play solution, easily integrated into existing LLMs without additional training costs." (Abstract)

## 11. Architectural Workarounds

- Hierarchical positional indexing across dimensions: "we use a h-dimensional vector to represent the hierarchical position index from high-level to low-level" and split dimensions to encode levels. (Section 3.1 Hierarchical format)
- Window mechanism for stability: "we also add a window mechanism, so that when dealing with short texts, our proposed method is consistent with the original positional encoding." (Section 3 Hierarchical RoPE)
- Function/class-level hierarchy: "Our hierarchical position includes both the token and function/class levels of the source code." (Section 5.5 Ablation Study)
- Alternative hierarchy splits (statement-level, fixed n-token segments): "we also try to split at the code statement level as well as implementing a strategy of splitting continuous n-tokens as a high-level segment (n = 128, 512, 1024)." (Section 5.5 Ablation Study)
- Natural language hierarchy proxy: "we set every 128 tokens as a segment, and encode it as higher-level position information." (Section 4 Experiment Setup)

## 12. Explicit Limitations and Non-Claims

- Limitations: "constrained by computational resources, we choose models below 7B for experiments." (Limitations)
- Limitations: "our discussion on the upper limit of the HiRoPE's performance tends to lean towards theoretical derivation." (Limitations)
- Limitations: "We are not clear whether some settings will implicitly affect the performance of the model." (Limitations)
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not specified.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: evaluates code and natural language text ("CodeParrot-valid dataset" and "ReRoPE-eval... curated from Common Crawl"). (Section 4 Experiment Setup)
> - Task structure: multiple defined tasks including language modeling, code symbol extraction, and long code completion ("language modeling"; "Code Symbol Understanding"; "long code completion benchmarks"). (Section 4 Experiment Setup)
> - Representation rigidity: hierarchical token positions with function/class levels and window length ("token and function/class levels"; "L_window = 512"). (Section 5.5 Ablation Study)
> - Model sharing vs specialization: plug-and-play without extra training ("without additional training costs"; "training-free solution"). (Abstract; Conclusion)
> - Role of positional encoding: central experimental variable with baselines and ablations ("enhancing the traditional rotary position embedding"; "including NTK... ReRoPE... Self-Extend"; "ablation studies"). (Contributions; Section 4.2 Baselines; Section 4 Experiment Setup)

### 14. Final Classification

Classification: **Multi-task, multi-domain (constrained)**

Justification: The paper evaluates multiple tasks including "language modeling" on CodeParrot, a new "Code Symbol Understanding" task, and "long code completion benchmarks" (LCC, RepoBench). (Section 4 Experiment Setup; Section 5.4 Long Code Completion) It also evaluates both code and natural language text, using "CodeParrot-valid dataset" and "ReRoPE-eval... curated from Common Crawl," which indicates multiple text domains rather than a single-domain setup. (Section 4 Experiment Setup)
