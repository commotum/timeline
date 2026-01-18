## 1. Basic Metadata

- Title: "YaRN: Efficient Context Window Extension of Large Language Models" (Title)
- Authors: "Bowen Peng" (Title block); "Jeffrey Quesnelle" (Title block); "Honglu Fan" (Title block); "Enrico Shippole" (Title block)
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

"We present YaRN (Yet another RoPE extensioN method), a compute-efficient method to extend the context window of such models, requiring 10x less tokens and 2.5x less training steps than previous methods." (Abstract)

## 3. Tasks Evaluated

Task 1
- Task name: Long sequence language modeling (perplexity)
- Task type: Generation
- Dataset(s) used: GovReport; Proof-pile
- Domain: Natural language long documents
- Evidence: "To evaluate the long sequence language modeling performances, we use the GovReport [18] and Proof-pile [4] datasets both of which contain many long sequence samples." (Section 4.3.1 Long Sequence Language Modeling)

Task 2
- Task name: Passkey retrieval
- Task type: Generation; Reasoning / relational; Other (specify: retrieval)
- Dataset(s) used: Not specified (task defined in [25])
- Domain: Natural language (synthetic/meaningless text)
- Evidence: "The passkey retrieval task as defined in [25] measures a model's ability to retrieve a simple passkey (i.e., a five-digit number) from amongst a large amount of otherwise meaningless text." (Section 4.3.2 Passkey Retrieval)

Task 3
- Task name: ARC-Challenge (25-shot)
- Task type: Classification; Reasoning / relational
- Dataset(s) used: ARC-Challenge
- Domain: Natural language multiple-choice QA
- Evidence: "Specifically, we use 25-shot ARC-Challenge [11], 10-shot HellaSwag [41], 5-shot MMLU [17], and 0-shot TruthfulQA [23]." (Section 4.3.3 Standardized Benchmarks)

Task 4
- Task name: HellaSwag (10-shot)
- Task type: Classification; Reasoning / relational
- Dataset(s) used: HellaSwag
- Domain: Natural language multiple-choice completion
- Evidence: "Specifically, we use 25-shot ARC-Challenge [11], 10-shot HellaSwag [41], 5-shot MMLU [17], and 0-shot TruthfulQA [23]." (Section 4.3.3 Standardized Benchmarks)

Task 5
- Task name: MMLU (5-shot)
- Task type: Classification; Reasoning / relational
- Dataset(s) used: MMLU
- Domain: Natural language multiple-choice QA
- Evidence: "Specifically, we use 25-shot ARC-Challenge [11], 10-shot HellaSwag [41], 5-shot MMLU [17], and 0-shot TruthfulQA [23]." (Section 4.3.3 Standardized Benchmarks)

Task 6
- Task name: TruthfulQA (0-shot)
- Task type: Classification; Reasoning / relational
- Dataset(s) used: TruthfulQA
- Domain: Natural language QA
- Evidence: "Specifically, we use 25-shot ARC-Challenge [11], 10-shot HellaSwag [41], 5-shot MMLU [17], and 0-shot TruthfulQA [23]." (Section 4.3.3 Standardized Benchmarks)

## 4. Domain and Modality Scope

- Evaluation performed on: Multiple domains within the same modality (text). Evidence: "To evaluate the long sequence language modeling performances, we use the GovReport [18] and Proof-pile [4] datasets both of which contain many long sequence samples." (Section 4.3.1 Long Sequence Language Modeling); "Specifically, we use 25-shot ARC-Challenge [11], 10-shot HellaSwag [41], 5-shot MMLU [17], and 0-shot TruthfulQA [23]." (Section 4.3.3 Standardized Benchmarks)
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Long sequence language modeling (perplexity) | Not specified. | Yes (context-extension fine-tuning). | Not specified (architecture unchanged beyond embedding frequencies). | "The evaluations focus on three aspects: - 1. the perplexity scores of fine-tuned models with extended context window, - 2. the passkey retrieval task on fine-tuned models, - 3. the common LLM benchmark results of fine-tuned models," (Section 4.3 Evaluation); "No changes were made to the LLaMA model architecture other than the calculation of the embedding frequencies as described in 3.4 with s=16 and s=32." (Section 4.1 Training) |
| Passkey retrieval | Not specified. | Yes (context-extension fine-tuning). | Not specified (architecture unchanged beyond embedding frequencies). | "The evaluations focus on three aspects: - 1. the perplexity scores of fine-tuned models with extended context window, - 2. the passkey retrieval task on fine-tuned models, - 3. the common LLM benchmark results of fine-tuned models," (Section 4.3 Evaluation); "No changes were made to the LLaMA model architecture other than the calculation of the embedding frequencies as described in 3.4 with s=16 and s=32." (Section 4.1 Training) |
| ARC-Challenge (25-shot) | Not specified. | Yes (context-extension fine-tuning). | Not specified (architecture unchanged beyond embedding frequencies). | "The evaluations focus on three aspects: - 1. the perplexity scores of fine-tuned models with extended context window, - 2. the passkey retrieval task on fine-tuned models, - 3. the common LLM benchmark results of fine-tuned models," (Section 4.3 Evaluation); "No changes were made to the LLaMA model architecture other than the calculation of the embedding frequencies as described in 3.4 with s=16 and s=32." (Section 4.1 Training) |
| HellaSwag (10-shot) | Not specified. | Yes (context-extension fine-tuning). | Not specified (architecture unchanged beyond embedding frequencies). | "The evaluations focus on three aspects: - 1. the perplexity scores of fine-tuned models with extended context window, - 2. the passkey retrieval task on fine-tuned models, - 3. the common LLM benchmark results of fine-tuned models," (Section 4.3 Evaluation); "No changes were made to the LLaMA model architecture other than the calculation of the embedding frequencies as described in 3.4 with s=16 and s=32." (Section 4.1 Training) |
| MMLU (5-shot) | Not specified. | Yes (context-extension fine-tuning). | Not specified (architecture unchanged beyond embedding frequencies). | "The evaluations focus on three aspects: - 1. the perplexity scores of fine-tuned models with extended context window, - 2. the passkey retrieval task on fine-tuned models, - 3. the common LLM benchmark results of fine-tuned models," (Section 4.3 Evaluation); "No changes were made to the LLaMA model architecture other than the calculation of the embedding frequencies as described in 3.4 with s=16 and s=32." (Section 4.1 Training) |
| TruthfulQA (0-shot) | Not specified. | Yes (context-extension fine-tuning). | Not specified (architecture unchanged beyond embedding frequencies). | "The evaluations focus on three aspects: - 1. the perplexity scores of fine-tuned models with extended context window, - 2. the passkey retrieval task on fine-tuned models, - 3. the common LLM benchmark results of fine-tuned models," (Section 4.3 Evaluation); "No changes were made to the LLaMA model architecture other than the calculation of the embedding frequencies as described in 3.4 with s=16 and s=32." (Section 4.1 Training) |

## 6. Input and Representation Constraints

- Fixed context length in pretraining: "As language models are usually pre-trained with a fixed context length, it is natural to ask how to extend the context length by fine-tuning on relatively less amount of data." (Section 2.2 Position Interpolation)
- Fixed-length training segments: "PG19 dataset [29] chunked into 64k segments bookended with the BOS and EOS token." (Section 4.1 Training)
- Variable evaluation sequence lengths: "evaluated the perplexity of each of these samples when truncated at 2k steps from a sequence length of 2k tokens through 128k tokens." (Section 4.3.1 Long Sequence Language Modeling)
- Embedding dimensionality constraint: "In RoPE, we first assume that |D| is even" (Section 2.1 Rotary Position Embeddings)
- Positional information dimensionality: "a token's positional information is one-dimensional" (Section 3.1 Loss of High Frequency information - "NTK-aware" interpolation)
- Padding/resizing requirements: Not specified.
- Fixed patch size or input resolution: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: "The models fine-tuned using YaRN has been made available and reproduced online up to 128k context length" (Abstract)
- Fixed vs variable sequence length: "In a lot of use cases, multiple forward-passes are performed with varying sequence lengths from 1 to the maximal context size." (Section 3.3 Dynamic Scaling - "Dynamic NTK" interpolation); "Throughout the whole inference cycle, the embedding layer is fixed including the scale factor s = L'/L where L' is the fixed number of extended context size." (Section 3.3 Dynamic Scaling - "Dynamic NTK" interpolation)
- Attention type: Not explicitly labeled; attention weights are defined as: "Next, the attention weights are calculated as

$$\operatorname{softmax}(\frac{\mathbf{q}_{m}^{T}\mathbf{k}_{n}}{\sqrt{|D|}}),\tag{2}$$" (Section 2.1 Rotary Position Embeddings)
- Mechanisms for computational cost: "kv-caching [8] is applied so that we can reuse the previous key-value vectors and improve the overall efficiency." (Section 3.3 Dynamic Scaling - "Dynamic NTK" interpolation); "For s=16 we fine-tuned for 400 steps with global batch size 64 using PyTorch [26] Fully Sharded Data Parallelism [42] and Flash Attention 2 [13]" (Section 4.1 Training)

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: "Rotary Position Embeddings (RoPE) have been shown to effectively encode positional information in transformer-based language models." (Abstract); "The basis of our work is the Rotary Position Embedding (RoPE) introduced in [34]." (Section 2.1 Rotary Position Embeddings)
- Where applied: "the attention layer first converts the vectors into the query vectors and the key vectors:" (Section 2.1 Rotary Position Embeddings)
- Fixed vs modified / compared: "By the "YaRN method", we refer to a combination of the attention scaling in Eq. 21 and the "NTK-by-parts" interpolation introduced in Section 3.2." (Section 3.4 YaRN); "Table 1 shows a side-by-side comparison of Llama-2 model extended from 4096 to 8192 context length via PI (LLongMA-2 7b<sup>5</sup>), "NTK-aware" and YaRN." (Section 4.3.1 Long Sequence Language Modeling)

## 9. Positional Encoding as a Variable

- Core research variable: "To this end, the position encodings of transformers are the center of the discussions." (Section 1 Introduction)
- Multiple positional encodings compared: "Table 1 shows a side-by-side comparison of Llama-2 model extended from 4096 to 8192 context length via PI (LLongMA-2 7b<sup>5</sup>), "NTK-aware" and YaRN." (Section 4.3.1 Long Sequence Language Modeling)
- Claim that PE choice is not critical or secondary: Not stated.

## 10. Evidence of Constraint Masking

- Model sizes: "For training, we extended the Llama 2 [39] 7B and 13B parameter models." (Section 4.1 Training)
- Dataset size / training steps: "only 400 training steps, representing approximately 0.1% of the model's original pre-training corpus" (Section 4 Experiments); "requiring 10x less tokens and 2.5x less training steps than previous methods." (Abstract); "For s=16 we fine-tuned for 400 steps with global batch size 64 using PyTorch [26] Fully Sharded Data Parallelism [42] and Flash Attention 2 [13] on the PG19 dataset [29] chunked into 64k segments bookended with the BOS and EOS token." (Section 4.1 Training)
- Attribution to method/training tricks: "We present YaRN (Yet another RoPE extensioN method), a compute-efficient method to extend the context window of such models" (Abstract); "By the "YaRN method", we refer to a combination of the attention scaling in Eq. 21 and the "NTK-by-parts" interpolation introduced in Section 3.2." (Section 3.4 YaRN)
- Architectural scaling vs method: "No changes were made to the LLaMA model architecture other than the calculation of the embedding frequencies as described in 3.4 with s=16 and s=32." (Section 4.1 Training)

## 11. Architectural Workarounds

- Dynamic Scaling (inference-time adaptation): "We call this inference-time method the Dynamic Scaling method." (Section 3.3 Dynamic Scaling - "Dynamic NTK" interpolation)
- Attention scaling in YaRN: "we modify the computation of attention weights into

$$\operatorname{softmax}\left(\frac{\mathbf{q}_{m}^{T}\mathbf{k}_{n}}{t\sqrt{|D|}}\right).$$" (Section 3.4 YaRN)
- Targeted RoPE interpolation to preserve local distances: "we choose not to interpolate the higher frequency dimensions at all while always interpolating the lower frequency dimensions." (Section 3.2 Loss of Relative Local Distances - "NTK-by-parts" interpolation)
- Efficiency technique: "kv-caching [8] is applied so that we can reuse the previous key-value vectors and improve the overall efficiency." (Section 3.3 Dynamic Scaling - "Dynamic NTK" interpolation)
- Sliding-window attention configuration (Mistral): "The model's sliding window attention size was set to the context window size, effectively disabling sliding window attention." (Section B.4 Mistral)

## 12. Explicit Limitations and Non-Claims

- Experimental limitation: "Due to compute constraints, we test only s=32 by further fine-tuning the s=16 model for 200 steps using the same dataset with 64k context." (Section 4.2 Extrapolation and Transfer Learning)
- Comparison scope limitation: "Since it is currently not compatible with Flash Attention 2 [13] and requires two attention passes during inference, we do not consider it for comparison." (Section 2.4 Related work)
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Text-only evaluation across multiple NLP datasets/benchmarks (GovReport, Proof-pile, ARC-Challenge, HellaSwag, MMLU, TruthfulQA).
> - Task structure: Language modeling perplexity + passkey retrieval + multiple-choice QA benchmarks.
> - Representation rigidity: Fixed pretraining context length; training uses fixed 64k-token segments; RoPE assumes even-dimensional embeddings and 1D positional information; dynamic scaling allows variable sequence lengths.
> - Model sharing vs specialization: Evaluations are on "fine-tuned models" for context extension with no architecture changes beyond embedding frequency calculation; no task-specific heads mentioned.
> - Role of positional encoding: Central variable; RoPE interpolation/attention scaling (YaRN) compared against PI and "NTK-aware" methods.

### 14. Final Classification

**Multi-task, single-domain**

The paper evaluates multiple tasks, including "the perplexity scores of fine-tuned models with extended context window," "the passkey retrieval task on fine-tuned models," and "the common LLM benchmark results of fine-tuned models" (Section 4.3 Evaluation), plus specific benchmarks like "ARC-Challenge" and "HellaSwag" (Section 4.3.3 Standardized Benchmarks). All evaluations are on text datasets such as GovReport and Proof-pile (Section 4.3.1 Long Sequence Language Modeling), so the experimental scope remains within a single modality/domain (natural language text).
