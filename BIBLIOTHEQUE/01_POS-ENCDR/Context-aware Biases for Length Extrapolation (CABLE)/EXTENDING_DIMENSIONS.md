## 1. Basic Metadata

- Title: "# Context-aware Biases for Length Extrapolation" (Title block)
- Authors: "# Ali Veisi\* Hamidreza Amirzadeh\* Amir M. Mansourian\* Algonet" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper proposes CABLE, "an additive RPE, Context-Aware Biases for Length Extrapolation (CABLE), a method that learns token-specific, context-aware biases for each attention head in transformers" to address length extrapolation (Abstract).

---

## 3. Tasks Evaluated

- Task name: Next-token prediction; Task type: Generation; Dataset(s): FineWeb-Edu-10B, WikiText-103, FineWeb (1B-token sample); Domain: natural language text (educational web pages, Wikipedia); Evidence: "We evaluate our proposed method on several benchmark datasets, using GPT-2 variants for next-token prediction and BERT models for long-context retrieval." (Contributions) "More specifically, we use a 10B sample of the FineWeb-Edu dataset, which consists of 1.3T tokens from educational web pages filtered from the FineWeb dataset." (Section 4.1 Datasets) "Furthermore, we also train the models on WikiText-103 (Merity et al., 2016), a small dataset containing a preprocessed version of Wikipedia, widely used in many NLP tasks." (Section 4.1 Datasets) "For evaluation, we use the test sets of FineWeb-Edu, WikiText-103, and a 1B-token sample of the FineWeb dataset." (Section 4.1 Datasets) "The evaluation metric is perplexity (PPL), and we train the models with sequence length of 1024." (Section 4.2 Settings)
- Task name: Masked language modeling (MLM) pretraining; Task type: Reconstruction; Dataset(s): FineWeb-Edu-10B; Domain: natural language text (educational web pages); Evidence: "We use the 10B-sample FineWeb-Edu dataset and train the models using only the masked language modeling (MLM) objective" (Section 5.5 Bidirectional Models) and "More specifically, we use a 10B sample of the FineWeb-Edu dataset, which consists of 1.3T tokens from educational web pages filtered from the FineWeb dataset." (Section 4.1 Datasets)
- Task name: Long-context retrieval; Task type: Other (retrieval); Dataset(s): MLDR (English subset) test set, MS-MARCO (fine-tuning); Domain: natural language text (long documents); Evidence: "we evaluate long-context performance using the English subset of MLDR (Chen et al., 2024), a retrieval benchmark consisting of over 200,000 long documents." (Section 5.5 Bidirectional Models) "To adapt BERT models for this task, we fine-tune them on MS-MARCO (Nguyen et al., 2016) using mined hard negatives (Xuan et al., 2020), with 1.25M samples, a batch size of 128, and a 5% learning rate warmup over one epoch" (Section 5.5 Bidirectional Models) "We then evaluate the fine-tuned models on the MLDR test set using nDCG@10 as the evaluation metric." (Section 5.5 Bidirectional Models)

---

## 4. Domain and Modality Scope

- Evaluation domain scope: Multiple domains within the same modality (text). Evidence: "More specifically, we use a 10B sample of the FineWeb-Edu dataset, which consists of 1.3T tokens from educational web pages filtered from the FineWeb dataset." (Section 4.1 Datasets) "Furthermore, we also train the models on WikiText-103 (Merity et al., 2016), a small dataset containing a preprocessed version of Wikipedia" (Section 4.1 Datasets) "we evaluate long-context performance using the English subset of MLDR (Chen et al., 2024), a retrieval benchmark consisting of over 200,000 long documents." (Section 5.5 Bidirectional Models)
- Domain generalization or cross-domain transfer claim: Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Next-token prediction (FineWeb-Edu-10B) | Not specified | Not specified | Not specified | "For all next-token prediction tasks, we use the GPT-2 variants (Brown et al., 2020). For the FineWeb-Edu-10B dataset, we use its small version (12 layers, 10 heads, and a hidden dimension of 768) with 124M parameters, and its medium version (24 layers, 16 heads, and a hidden dimension of 1024) with 334M parameters." (Section 4.2 Settings) |
| Next-token prediction (WikiText-103) | Not specified | Not specified | Not specified | "For all next-token prediction tasks, we use the GPT-2 variants (Brown et al., 2020)." (Section 4.2 Settings) "Furthermore, we also train the models on WikiText-103 (Merity et al., 2016), a small dataset containing a preprocessed version of Wikipedia, widely used in many NLP tasks." (Section 4.1 Datasets) |
| Masked language modeling (MLM) pretraining | Not specified | Not specified | Not specified | "We use the 10B-sample FineWeb-Edu dataset and train the models using only the masked language modeling (MLM) objective" (Section 5.5 Bidirectional Models) |
| Long-context retrieval (MLDR) | Not specified | Yes | Not specified | "To adapt BERT models for this task, we fine-tune them on MS-MARCO (Nguyen et al., 2016) using mined hard negatives (Xuan et al., 2020), with 1.25M samples" (Section 5.5 Bidirectional Models) "We then evaluate the fine-tuned models on the MLDR test set using nDCG@10 as the evaluation metric." (Section 5.5 Bidirectional Models) |

---

## 6. Input and Representation Constraints

- Fixed training sequence length for GPT-2 LM: "The evaluation metric is perplexity (PPL), and we train the models with sequence length of 1024." (Section 4.2 Settings)
- Fixed maximum sequence length for BERT pretraining: "Our BERT models are based on the bert-base-uncased architecture and are trained on four H100 GPUs with a batch size of 32 and a maximum sequence length of 512 for 14k steps" (Section 5.5 Bidirectional Models)
- Retrieval training length with longer evaluation: "Table 3: Retrieval performance (nDCG@10) on the MLDR test set for BERT models with different positional encodings, trained at sequence length 512 and evaluated on longer inputs." (Table 3 caption)
- Fixed number of positions for Learnable APE baseline: "The number of positions is fixed and predefined during training." (Section 4.3 Baselines)
- Representation dimensionality per head: "Let t and d be the sequence length and the dimension of embeddings on each head, respectively." (Section 3 Proposed Method)
- Padding or resizing requirements: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length reported: "For the longest sequences tested, we report results for 15,360 tokens instead of 16,384 due to computational constraints." (Section 5.1 Length Extrapolation) and "16384" (Table 3)
- Fixed or variable sequence length: "trained with a sequence length of 1024 and evaluated on shorter and longer sequences" (Section 5.1 Length Extrapolation) and "trained at sequence length 512 and evaluated on longer inputs." (Table 3 caption)
- Attention type: Not specified; the paper notes only that "training is typically performed on short sequences to mitigate the quadratic cost of attention." (Section 1 Introduction)
- Mechanisms to manage cost: "training is typically performed on short sequences to mitigate the quadratic cost of attention." (Section 1 Introduction) "it can be trained on shorter sequence lengths and effectively tested on much longer sequences" (Section 5.2 Runtime and Memory Overhead) and "for inference, we cache the cumulative sums, so there is no need to re-calculate them for all tokens each time." (Section 5.2 Runtime and Memory Overhead)

---

## 8. Positional Encoding (Critical Section)

- Mechanism: Bias-based additive relative positional encoding; "CABLE computes context-aware positional bias scores for each attention head and adds them to the pre-softmax attention logits." (Section 3 Proposed Method) and "a novel additive relative positional encoding (RPE) approach" (Section 3 Proposed Method)
- Where applied: "As with most RPE methods, CABLE adds positional information only to the queries and keys (not the values), a practice shown to enhance length extrapolation in methods like ALiBi, T5-bias, and RoPE." (Section 3 Proposed Method) and "injecting them into the attention matrix at every decoder layer." (Section 6 Conclusion)
- Fixed or varied across experiments: Compared and ablated; "In our experiments, we refer to this version—without any additional learnable parameters for the biases—as CABLE<sub>NW</sub>." (Section 3 Proposed Method) "We compare our method against the following positional encoding approaches:" (Section 4.3 Baselines) and "we tested a kernelized version of our method (K-CABLE)." (Section 5.6 Ablation Study)

---

## 9. Positional Encoding as a Variable

- Core research variable: "We propose CABLE, an additive relative positional encoding method" (Contributions)
- Multiple positional encodings compared: "We compare our method against the following positional encoding approaches:" (Section 4.3 Baselines)
- Claim that PE choice is not critical or secondary: Not stated.

---

## 10. Evidence of Constraint Masking

- Model sizes: "GPT-2 Medium (334M parameters)" (Abstract) "For the FineWeb-Edu-10B dataset, we use its small version (12 layers, 10 heads, and a hidden dimension of 768) with 124M parameters, and its medium version (24 layers, 16 heads, and a hidden dimension of 1024) with 334M parameters." (Section 4.2 Settings) and "We also incorporate a tiny version of GPT-2 (44M parameters) with 6 layers, 8 heads, and a hidden dimension of 512" (Section 4.2 Settings)
- Dataset sizes: "For training, we use the FineWeb dataset (Penedo et al., 2024), a large-scale dataset (15 trillion tokens) for LLM pretraining, derived from 96 CommonCrawl snapshots." (Section 4.1 Datasets) "More specifically, we use a 10B sample of the FineWeb-Edu dataset, which consists of 1.3T tokens from educational web pages filtered from the FineWeb dataset." (Section 4.1 Datasets) "We allocate 9.9B tokens for training and 0.1B for evaluation." (Section 4.1 Datasets) "a retrieval benchmark consisting of over 200,000 long documents." (Section 5.5 Bidirectional Models) and "with 1.25M samples" (Section 5.5 Bidirectional Models)
- Attribution of gains: The paper attributes gains to the positional-bias mechanism rather than scaling; "CABLE computes context-aware positional bias scores for each attention head and adds them to the pre-softmax attention logits." (Section 3 Proposed Method) and "Experiments show that CABLE lowers perplexity, significantly improves length extrapolation, and consistently outperforms baselines" (Section 6 Conclusion). No explicit claim that gains primarily come from scaling model size or data.

---

## 11. Architectural Workarounds

- Lightweight additive bias design: "It requires only two additional linear layers, minimal parameters, and can be implemented in a few lines of code. The design involves two unfolding operations, a cumulative summation, and bias addition to the attention logits." (Section 3 Proposed Method)
- Sliding-window-like inductive bias: "CABLE exhibits an inductive bias similar to sliding window attention by penalizing distant querykey pairs" (Section 3 Proposed Method)
- Inference optimization: "for inference, we cache the cumulative sums, so there is no need to re-calculate them for all tokens each time." (Section 5.2 Runtime and Memory Overhead)
- Train-short/test-long strategy enabled by PE: "it can be trained on shorter sequence lengths and effectively tested on much longer sequences." (Section 5.2 Runtime and Memory Overhead)

---

## 12. Explicit Limitations and Non-Claims

- Limitations: "it incurs higher training time compared to RoPE due to its dynamic bias computation, though this overhead is negligible in inference." (Limitation) "CABLE occasionally underperforms RoPE at base sequence lengths (e.g., 1024 tokens in our experiments)" (Limitation)
- Further constraints and future work: "the method's computational overhead, though minimal, may become more pronounced for extremely long sequences (>100K tokens), and its extrapolation capabilities remain dependent on the diversity of positional patterns in training data. While empirical results are promising, theoretical analysis of its attention dynamics at arbitrary lengths remains an open question. Future work could explore optimizations for training efficiency and head-specific bias adaptation to further enhance flexibility." (Limitation)
- Explicit non-claims about open-world learning or unrestrained multi-task learning: Not stated.

---

### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: Multiple text datasets (web pages, Wikipedia, long documents) within a single modality.
- Task structure: Next-token prediction and long-context retrieval are both evaluated, with MLM used for BERT pretraining.
- Representation rigidity: Fixed training lengths (1024 for GPT-2, 512 for BERT) with evaluation on longer sequences.
- Model sharing vs specialization: GPT-2 variants are described per dataset; BERT is pre-trained then fine-tuned for retrieval.
- Role of positional encoding: Central variable with multiple PE baselines plus CABLE variants/ablations.

---

### 14. Final Classification

Multi-task, single-domain.

The paper evaluates multiple tasks, including "next-token prediction" and "long-context retrieval" (Contributions), indicating a multi-task setup. All evaluations are on natural language text datasets such as "educational web pages" (FineWeb-Edu), "preprocessed version of Wikipedia" (WikiText-103), and "long documents" (MLDR), so the modality/domain is single (text) (Sections 4.1 and 5.5).
