## 1. Basic Metadata

- Title: "EXTENDING CONTEXT WINDOW OF LARGE LANGUAGE MODELS VIA POSITION INTERPOLATION" (Title)
- Authors: "Shouyuan Chen Sherman Wong Liangjian Chen Yuandong Tian Meta Platforms Inc." (Top of document)
- Year: Year not specified.
- Venue: Venue not specified.

---

## 2. One-Sentence Contribution Summary

"We present Position Interpolation (PI) that extends the context window sizes of RoPE-based (Su et al., 2021) pretrained LLMs such as LLaMA (Touvron et al., 2023) models to up to 32768 with minimal fine-tuning (within 1000 steps), while demonstrating strong empirical results on various tasks that require long context, including passkey retrieval, language modeling, and long document summarization from LLaMA 7B to 65B." (Abstract)

---

## 3. Tasks Evaluated

- Task name: Long sequence language modeling
  - Task type: Generation
  - Dataset(s) used: "book corpus (PG-19)"; "cleaned Arxiv Math proof-pile dataset" (3.2 Long Sequence Language Modeling)
  - Domain: Natural language text (books and math proofs) ("long sequence language modeling"; 3.2 Long Sequence Language Modeling)
  - Evidence: "We evaluate the long sequence language modeling performance of our extended models and baselines on two datasets: book corpus (PG-19) (Rae et al., 2020) and cleaned Arxiv Math proof-pile dataset (Azerbayev et al., 2022)." (3.2 Long Sequence Language Modeling)

- Task name: Passkey retrieval
  - Task type: Other (passkey retrieval)
  - Dataset(s) used: Synthetic passkey retrieval task (Mohtashami & Jaggi, 2023)
  - Domain: Natural language text (synthetic long document) ("long document"; 3.3 Measuring Effective Context Window Size through Passkey Retrieval)
  - Evidence: "we follow a synthetic evaluation task of passkey retrieval proposed by Mohtashami & Jaggi (2023). In this task, the models are asked to recover a random passkey hidden in a long document." (3.3 Measuring Effective Context Window Size through Passkey Retrieval)

- Task name: Long document summarization
  - Task type: Generation
  - Dataset(s) used: "GovReport" (Huang et al., 2021)
  - Domain: Natural language text (long documents) ("long document summarization"; 3.5 Long Document Summarization)
  - Evidence: "In this task, we evaluate our models' performance on the long document summarization task. In particular, we consider the GovReport (Huang et al., 2021) dataset, which contains 17457 documents for training and 972 documents for evaluation." (3.5 Long Document Summarization)

- Task name: BoolQ
  - Task type: Classification
  - Dataset(s) used: BoolQ
  - Domain: Not explicitly stated (benchmark names only)
  - Evidence: "Table 5: Zero-shot performance on a subset of LLaMA Benchmarks." and the table header "BoolQ | PIQA | Race-M | Race-H | WinoGrande" (Table 5)

- Task name: PIQA
  - Task type: Classification
  - Dataset(s) used: PIQA
  - Domain: Not explicitly stated (benchmark names only)
  - Evidence: "Table 5: Zero-shot performance on a subset of LLaMA Benchmarks." and the table header "BoolQ | PIQA | Race-M | Race-H | WinoGrande" (Table 5)

- Task name: Race-M
  - Task type: Classification
  - Dataset(s) used: Race-M
  - Domain: Not explicitly stated (benchmark names only)
  - Evidence: "Table 5: Zero-shot performance on a subset of LLaMA Benchmarks." and the table header "BoolQ | PIQA | Race-M | Race-H | WinoGrande" (Table 5)

- Task name: Race-H
  - Task type: Classification
  - Dataset(s) used: Race-H
  - Domain: Not explicitly stated (benchmark names only)
  - Evidence: "Table 5: Zero-shot performance on a subset of LLaMA Benchmarks." and the table header "BoolQ | PIQA | Race-M | Race-H | WinoGrande" (Table 5)

- Task name: WinoGrande
  - Task type: Classification
  - Dataset(s) used: WinoGrande
  - Domain: Not explicitly stated (benchmark names only)
  - Evidence: "Table 5: Zero-shot performance on a subset of LLaMA Benchmarks." and the table header "BoolQ | PIQA | Race-M | Race-H | WinoGrande" (Table 5)

---

## 4. Domain and Modality Scope

- Evaluation is performed on a single modality/domain (natural language text), with tasks including "language modeling," "passkey retrieval," and "long document summarization." (Abstract)
- Multiple domains within the same modality are present as multiple text datasets (e.g., "book corpus (PG-19)" and "Arxiv Math proof-pile dataset"). (3.2 Long Sequence Language Modeling)
- Domain generalization or cross-domain transfer: Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Long sequence language modeling | Yes (same extended LLaMA variants evaluated) | Yes (context-window fine-tuning) | No (architecture unchanged) | "We evaluate the long sequence language modeling performance of our extended models and baselines" (3.2 Long Sequence Language Modeling); "We fine-tune all model variants using the next token prediction objective." (3.1 Setup); "we did not modify LLaMA model architectures" (3.1 Setup) |
| Passkey retrieval | Yes (same extended LLaMA variants evaluated) | Yes (context-window fine-tuning) | No (architecture unchanged) | "We evaluate the 7B and 33B LLaMA model variants that are extended via Position Interpolation or direct fine-tuning." (3.3 Measuring Effective Context Window Size through Passkey Retrieval); "we did not modify LLaMA model architectures" (3.1 Setup) |
| Long document summarization | No (task-specific fine-tuning) | Yes (task-specific fine-tuning) | No (architecture unchanged) | "We fine-tune the LLaMA models extended with Position Interpolation with a context window of 16384." (3.5 Long Document Summarization); "we did not modify LLaMA model architectures" (3.1 Setup) |
| BoolQ | Yes (zero-shot on extended models) | No (zero-shot for this task) | No (architecture unchanged) | "Table 5: Zero-shot performance on a subset of LLaMA Benchmarks." (Table 5); "we did not modify LLaMA model architectures" (3.1 Setup) |
| PIQA | Yes (zero-shot on extended models) | No (zero-shot for this task) | No (architecture unchanged) | "Table 5: Zero-shot performance on a subset of LLaMA Benchmarks." (Table 5); "we did not modify LLaMA model architectures" (3.1 Setup) |
| Race-M | Yes (zero-shot on extended models) | No (zero-shot for this task) | No (architecture unchanged) | "Table 5: Zero-shot performance on a subset of LLaMA Benchmarks." (Table 5); "we did not modify LLaMA model architectures" (3.1 Setup) |
| Race-H | Yes (zero-shot on extended models) | No (zero-shot for this task) | No (architecture unchanged) | "Table 5: Zero-shot performance on a subset of LLaMA Benchmarks." (Table 5); "we did not modify LLaMA model architectures" (3.1 Setup) |
| WinoGrande | Yes (zero-shot on extended models) | No (zero-shot for this task) | No (architecture unchanged) | "Table 5: Zero-shot performance on a subset of LLaMA Benchmarks." (Table 5); "we did not modify LLaMA model architectures" (3.1 Setup) |

---

## 6. Input and Representation Constraints

- Fixed original context limit: "inputs to LLaMA models (Touvron et al., 2023) must be fewer than 2048 tokens." (1 Introduction)
- Extended maximum context length: "extend the context window to up to 32768 from the initial 2048" (3 Experiments)
- Positional index rescaling constraint: "we reduce position indices from [0, L') to [0, L) to match the original range of indices before computing RoPE." (2.3 Proposed Approach: Position Interpolation)
- Proof-pile truncation: "truncate to the first 32768 tokens for each test document." (3.2 Long Sequence Language Modeling)
- Summarization truncation: "We truncate all input documents to their first 15000 tokens." (3.5 Long Document Summarization)
- Summarization output cap: "The final output is truncated at 1000 tokens." (3.5 Long Document Summarization)

---

## 7. Context Window and Attention Structure

- Maximum sequence length: "extend the context window to up to 32768" (3 Experiments)
- Sequence length fixed or variable: A pre-defined window limit is imposed ("must be fewer than 2048 tokens"), and models are extended to fixed larger windows ("context window of sizes up to 32768"). (1 Introduction; 3.1 Setup)
- Attention type: Full/global attention with unmodified mechanism ("with an unmodified attention mechanism and model architecture"; "Our work allows full access of the entire input through unmodified attention"). (4 Related Work)
- Mechanisms to manage computational cost: No architectural changes ("we did not modify LLaMA model architectures"), but training uses "Flash Attention" for efficiency. (3.1 Setup)

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: RoPE (Rotary Position Embedding): "We consider Rotary Position Embedding (RoPE) (Su et al., 2021), which is the position encoding used in the LLaMA model (Touvron et al., 2023)." (2.1 Background: Rotary Position Embedding)
- Where applied: "At each layer, RoPE is applied on both query and key embeddings for computing attention scores." (2.1 Background: Rotary Position Embedding)
- Modification for PI: "we replace RoPE f by f' defined as follows" and "we reduce position indices from [0, L') to [0, L) to match the original range of indices before computing RoPE." (2.3 Proposed Approach: Position Interpolation)
- Fixed vs modified across experiments: Compared across methods ("using either direct fine-tuning or Position Interpolation method"). (3.1 Setup)

---

## 9. Positional Encoding as a Variable

- Core research variable: Yes; the paper introduces and studies PI vs extrapolation ("DIRECT EXTRAPOLATION" vs "PROPOSED APPROACH: POSITION INTERPOLATION (PI)"). (2.2; 2.3)
- Multiple positional encodings compared: Yes, interpolation vs direct extrapolation/fine-tuning ("using either direct fine-tuning or Position Interpolation method"). (3.1 Setup)
- Claim that PE choice is not critical or secondary: Not claimed.

---

## 10. Evidence of Constraint Masking

- Model sizes: "we extended the pre-trained 7B, 13B, 33B and 65B LLaMA models" (3.1 Setup)
- Dataset sizes: "PG19 ... test split consisting of 100 documents"; "proof-pile ... random subsample of 128 documents"; "GovReport ... 17457 documents for training and 972 documents for evaluation." (3.2 Long Sequence Language Modeling; 3.5 Long Document Summarization)
- Performance gains attributed to interpolation (not model/data scaling): "Position Interpolation can easily enable very long context windows (e.g. 32768), requiring only fine-tuning for 1000 steps on the Pile" and "Position Interpolation generates strong models that can effectively make use of much extended context window." (Abstract)
- Direct fine-tuning is reported as ineffective by comparison: "models ... extended via direct fine-tuning only saw a minimal increase of the effective context window size ... even after fine-tuning for more than 10000 steps." (3.3 Measuring Effective Context Window Size through Passkey Retrieval)

---

## 11. Architectural Workarounds

- No architectural changes or specialized attention: "Except for rescaling the position indices for models extended with Position Interpolation, we did not modify LLaMA model architectures (Touvron et al., 2023) in any ways." (3.1 Setup)
- No added parameters: "our method of rescaling of position indices does not introduce extra weight, or modify the model architecture in any way." (2.3 Proposed Approach: Position Interpolation)

---

## 12. Explicit Limitations and Non-Claims

- Higher inference cost: "Our work allows attending to all previous tokens, preserving all details without compression, albeit with higher inference costs." (4 Related Work)
- Future work on regularization: "we are not aware of existing LLM pre-training techniques that leverage this regularization and will leave it for future work." (2.3 Proposed Approach: Position Interpolation)
- Future work on other position encodings: "we plan to investigate in such directions in the near future." (5 Conclusions)
- Non-claim about attention approximations: "Although not the focus of this work, as these methods are not used in LLaMA (Touvron et al., 2023)." (4 Related Work)

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: single text modality with tasks like "language modeling," "passkey retrieval," and "long document summarization." (Abstract)
> – Task structure: multiple evaluation tasks, including "long sequence language modeling" and "long document summarization," plus zero-shot benchmarks in Table 5. (3.2 Long Sequence Language Modeling; 3.5 Long Document Summarization; Table 5)
> – Representation rigidity: fixed context limits and truncation ("must be fewer than 2048 tokens"; "truncate to the first 32768 tokens"; "truncate all input documents to their first 15000 tokens"). (1 Introduction; 3.2; 3.5)
> – Model sharing vs specialization: extended LLaMA variants are evaluated across tasks, but summarization uses task-specific fine-tuning ("We fine-tune the LLaMA models extended with Position Interpolation" for summarization). (3.5 Long Document Summarization)
> – Role of positional encoding: central variable ("Rotary Position Embedding (RoPE)" and the PI rescaling of indices). (2.1; 2.3)

---

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple tasks ("language modeling, and long document summarization" plus "passkey retrieval" and zero-shot benchmark tasks in Table 5), indicating multi-task evaluation. (Abstract; 3.3; Table 5) All evaluations are within text-only settings (e.g., "long sequence language modeling" on text datasets and "random passkey hidden in a long document"), and no cross-domain transfer is claimed. (3.2; 3.3)
