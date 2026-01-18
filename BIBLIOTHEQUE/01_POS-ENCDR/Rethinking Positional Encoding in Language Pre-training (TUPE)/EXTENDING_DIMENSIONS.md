## 1. Basic Metadata

- Title: RETHINKING POSITIONAL ENCODING IN LANGUAGE PRE-TRAINING. Evidence: "RETHINKING POSITIONAL ENCODING IN LANGUAGE PRE-TRAINING" (Title).
- Authors: Guolin Ke, Di He & Tie-Yan Liu. Evidence: "Guolin Ke, Di He & Tie-Yan Liu" (Title).
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper states, "we investigate the positional encoding methods used in language pretraining (e.g., BERT) and identify several problems in the existing formulations" and "Motivated from above analysis, we propose a new positional encoding method called Transformer with Untied Positional Encoding (TUPE)" (ABSTRACT).

## 3. Tasks Evaluated

| Task name | Task type | Dataset(s) used | Domain | Evidence |
| --- | --- | --- | --- | --- |
| CoLA | Other (task type not specified in text) | "the GLUE (General Language Understanding Evaluation) dataset" (Section 4.1 EXPERIMENTAL DESIGN) | "language pretraining" (ABSTRACT) | "we use nine tasks in GLUE, including CoLA, RTE, MRPC, STS, SST, QNLI, QQP, and MNLI-m/mm." (Section B EXPERIMENTAL DETAILS) |
| RTE | Other (task type not specified in text) | "the GLUE (General Language Understanding Evaluation) dataset" (Section 4.1 EXPERIMENTAL DESIGN) | "language pretraining" (ABSTRACT) | "we use nine tasks in GLUE, including CoLA, RTE, MRPC, STS, SST, QNLI, QQP, and MNLI-m/mm." (Section B EXPERIMENTAL DETAILS) |
| MRPC | Other (task type not specified in text) | "the GLUE (General Language Understanding Evaluation) dataset" (Section 4.1 EXPERIMENTAL DESIGN) | "language pretraining" (ABSTRACT) | "we use nine tasks in GLUE, including CoLA, RTE, MRPC, STS, SST, QNLI, QQP, and MNLI-m/mm." (Section B EXPERIMENTAL DETAILS) |
| STS | Other (task type not specified in text) | "the GLUE (General Language Understanding Evaluation) dataset" (Section 4.1 EXPERIMENTAL DESIGN) | "language pretraining" (ABSTRACT) | "we use nine tasks in GLUE, including CoLA, RTE, MRPC, STS, SST, QNLI, QQP, and MNLI-m/mm." (Section B EXPERIMENTAL DETAILS) |
| SST | Other (task type not specified in text) | "the GLUE (General Language Understanding Evaluation) dataset" (Section 4.1 EXPERIMENTAL DESIGN) | "language pretraining" (ABSTRACT) | "we use nine tasks in GLUE, including CoLA, RTE, MRPC, STS, SST, QNLI, QQP, and MNLI-m/mm." (Section B EXPERIMENTAL DETAILS) |
| QNLI | Other (task type not specified in text) | "the GLUE (General Language Understanding Evaluation) dataset" (Section 4.1 EXPERIMENTAL DESIGN) | "language pretraining" (ABSTRACT) | "we use nine tasks in GLUE, including CoLA, RTE, MRPC, STS, SST, QNLI, QQP, and MNLI-m/mm." (Section B EXPERIMENTAL DETAILS) |
| QQP | Other (task type not specified in text) | "the GLUE (General Language Understanding Evaluation) dataset" (Section 4.1 EXPERIMENTAL DESIGN) | "language pretraining" (ABSTRACT) | "we use nine tasks in GLUE, including CoLA, RTE, MRPC, STS, SST, QNLI, QQP, and MNLI-m/mm." (Section B EXPERIMENTAL DETAILS) |
| MNLI-m | Other (task type not specified in text) | "the GLUE (General Language Understanding Evaluation) dataset" (Section 4.1 EXPERIMENTAL DESIGN) | "language pretraining" (ABSTRACT) | "we use nine tasks in GLUE, including CoLA, RTE, MRPC, STS, SST, QNLI, QQP, and MNLI-m/mm." (Section B EXPERIMENTAL DETAILS) |
| MNLI-mm | Other (task type not specified in text) | "the GLUE (General Language Understanding Evaluation) dataset" (Section 4.1 EXPERIMENTAL DESIGN) | "language pretraining" (ABSTRACT) | "we use nine tasks in GLUE, including CoLA, RTE, MRPC, STS, SST, QNLI, QQP, and MNLI-m/mm." (Section B EXPERIMENTAL DETAILS) |

## 4. Domain and Modality Scope

- Single domain? Yes, language/text. Evidence: "language pretraining" (ABSTRACT); "We use the GLUE (General Language Understanding Evaluation) dataset (Wang et al., 2018) as the downstream tasks" (Section 4.1 EXPERIMENTAL DESIGN).
- Multiple domains within the same modality? Not stated.
- Multiple modalities? Not stated.
- Domain generalization or cross-domain transfer claim? Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| CoLA | Shared pre-trained initialization; fine-tuned per task | Yes | Not specified | "Following previous works, we search the learning rates during the fine-tuning for each downstream task." (Section B EXPERIMENTAL DETAILS) |
| RTE | Shared pre-trained initialization; fine-tuned per task | Yes | Not specified | "Following previous works, we search the learning rates during the fine-tuning for each downstream task." (Section B EXPERIMENTAL DETAILS) |
| MRPC | Shared pre-trained initialization; fine-tuned per task | Yes | Not specified | "Following previous works, we search the learning rates during the fine-tuning for each downstream task." (Section B EXPERIMENTAL DETAILS) |
| STS | Shared pre-trained initialization; fine-tuned per task | Yes | Not specified | "Following previous works, we search the learning rates during the fine-tuning for each downstream task." (Section B EXPERIMENTAL DETAILS) |
| SST | Shared pre-trained initialization; fine-tuned per task | Yes | Not specified | "Following previous works, we search the learning rates during the fine-tuning for each downstream task." (Section B EXPERIMENTAL DETAILS) |
| QNLI | Shared pre-trained initialization; fine-tuned per task | Yes | Not specified | "Following previous works, we search the learning rates during the fine-tuning for each downstream task." (Section B EXPERIMENTAL DETAILS) |
| QQP | Shared pre-trained initialization; fine-tuned per task | Yes | Not specified | "Following previous works, we search the learning rates during the fine-tuning for each downstream task." (Section B EXPERIMENTAL DETAILS) |
| MNLI-m | Shared pre-trained initialization; fine-tuned per task | Yes | Not specified | "Following previous works, we search the learning rates during the fine-tuning for each downstream task." (Section B EXPERIMENTAL DETAILS) |
| MNLI-mm | Shared pre-trained initialization; fine-tuned per task | Yes | Not specified | "Following previous works, we search the learning rates during the fine-tuning for each downstream task." (Section B EXPERIMENTAL DETAILS) |

## 6. Input and Representation Constraints

- Maximum sequence length: "the maximum sequence length is 512" (Section B EXPERIMENTAL DETAILS).
- Sequence length setting (pre-training and fine-tuning): "Sequence Length 512" (Table 2: Hyperparameters for the pre-training and fine-tuning).
- Tokenization and vocabulary size: "We set the vocabulary size (sub-word tokens) as 32,768." (Section 4.1 EXPERIMENTAL DESIGN); "applying byte pair encoding (BPE) (Sennrich et al., 2015) with setting the vocabulary size as 32,768." (Section B EXPERIMENTAL DETAILS).
- Input preparation: "segmenting documents into sentences by Spacy<sup>5</sup>, normalizing, lower-casing, and tokenizing the texts by Moses decoder (Koehn et al., 2007)" (Section B EXPERIMENTAL DETAILS).
- Sentence packing: "We remove the next sentence prediction task and use FULL-SENTENCES mode to pack sentences" (Section B EXPERIMENTAL DETAILS).
- Fixed/variable input resolution, fixed patch size, fixed dimensionality, padding/resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: "the maximum sequence length is 512" (Section B EXPERIMENTAL DETAILS).
- Fixed vs variable length: Not explicitly stated; only a maximum is given ("maximum sequence length is 512" in Section B EXPERIMENTAL DETAILS).
- Attention type: Global self-attention over all positions, as in "z_i^l = \sum_{j=1}^n" (Section 2.1 Attention module).
- Computational cost management: "For efficiency, we share the (multi-head) projection matrices ${\cal U}^Q$ and ${\cal U}^K$ in different layers." and "we only need to compute it in the first layer, and reuse its outputs in other layers." (IMPLEMENTATION DETAILS AND DISCUSSIONS).

## 8. Positional Encoding (Critical Section)

- Mechanism used: Absolute and relative positional encodings are described, e.g., "a (learnable) real-valued vector  $p_i$  is assigned to each position i" and "$a_{j-i}^l$  is learnable parameter" with a bias term "$b_{j-i}$" (Section 2.2 Positional Encoding). TUPE uses untied positional correlations: "In TUPE, the Transformer only uses the word embedding as input. In the self-attention module, different types of correlations are separately computed to reflect different aspects of information, including word contextual correlation and absolute (and relative) positional correlation." (Section 1 Introduction).
- Where applied: TUPE removes positional embeddings from the input and applies positional correlations inside attention: "we first remove the absolute positional encoding from the input of the Transformer and compute the positional correlation and word correlation separately with different projection matrices in the self-attention module." (Section 6 CONCLUSION).
- Fixed vs modified/ablated: Multiple positional encodings are compared and ablated, e.g., "There are two versions of TUPE." and "We call them TUPE-A and TUPE-R respectively" (IMPLEMENTATION DETAILS AND DISCUSSIONS) and "To compare with TUPE-A and TUPE-R, we set up two baselines correspondingly: BERT-A, which is the standard BERT-Base with absolute positional encoding (Devlin et al., 2018); BERT-R, which uses both absolute positional encoding and relative positional encoding (Raffel et al., 2019) (Eq. (5))." (Section 4.1 EXPERIMENTAL DESIGN).

## 9. Positional Encoding as a Variable

- Core research variable? Yes: "we investigate the positional encoding methods used in language pretraining (e.g., BERT) and identify several problems in the existing formulations" and "we propose a new positional encoding method called Transformer with Untied Positional Encoding (TUPE)" (ABSTRACT).
- Multiple positional encodings compared? Yes: "There are two versions of TUPE." and "We call them TUPE-A and TUPE-R respectively" (IMPLEMENTATION DETAILS AND DISCUSSIONS) and "To compare with TUPE-A and TUPE-R, we set up two baselines correspondingly: BERT-A, which is the standard BERT-Base with absolute positional encoding (Devlin et al., 2018); BERT-R, which uses both absolute positional encoding and relative positional encoding (Raffel et al., 2019) (Eq. (5))." (Section 4.1 EXPERIMENTAL DESIGN).
- PE choice claimed to be not critical/secondary? Not claimed.

## 10. Evidence of Constraint Masking

- Model sizes: "We use BERT-Base (110M parameters) architecture for all experiments." (Section 4.1 EXPERIMENTAL DESIGN); "Different models are pre-trained in the BERT-Large setting (330M)" (Table 3 caption); "Different models are pre-trained in the ELECTRA-Base setting (120M)" (Table 4 caption).
- Dataset size: "By concatenating these two datasets, we obtain a corpus with roughly 16GB in size." (Section 4.1 EXPERIMENTAL DESIGN).
- Attribution of gains: "As the only difference between TUPE and baselines is the positional encoding, these results indicate TUPE can better utilize the positional information in sequence." (Section 4.2 OVERALL COMPARISON) and "with a better inductive bias over the positional information, TUPE can even outperform the baselines while only using 30% pre-training computational costs." (Section 6 CONCLUSION).

## 11. Architectural Workarounds

- Untied positional and word correlations in attention: "we propose to directly model the relationships between a pair of words or positions by using different projection matrices and remove the two terms in the middle." (Section 3.1 Until the Correlations between Positions and Words).
- Untying [CLS] positional correlations: "We give a specific design in the attention module to untie the [CLS] symbol from other positions." (Section 3.2 Untie the [CLS] Symbol from Positions).
- Efficiency via sharing and reuse: "For efficiency, we share the (multi-head) projection matrices ${\cal U}^Q$ and ${\cal U}^K$ in different layers." and "we only need to compute it in the first layer, and reuse its outputs in other layers." (IMPLEMENTATION DETAILS AND DISCUSSIONS).
- Relative positional bias term: "+ b_{j-i}." (Equation (8), Section 3.1).

## 12. Explicit Limitations and Non-Claims

- Failed attempts: "We tried to replace the parametric form of the positional correlation  $(\frac{1}{\sqrt{2d}}(p_iU^Q)(p_jU^K)^T)$  to the non-parametric form." (Section D FAILED ATTEMPTS); "However, empirically we found that the training of this setting converges much slower than the baselines." (Section D FAILED ATTEMPTS).
- Failed attempts: "We also tried to parameterize relative position bias  $b_{j-i}$  by  $(r_{j-i}F^Q)(r_{j-i}F^K)^T$ . But the improvement is just marginal." (Section D FAILED ATTEMPTS).
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Single-domain language/text evaluation ("language pretraining" in ABSTRACT; GLUE downstream tasks in Section 4.1 EXPERIMENTAL DESIGN).
> - Task structure: Multiple supervised downstream tasks within GLUE (Section B EXPERIMENTAL DETAILS).
> - Representation rigidity: Fixed maximum sequence length of 512 and fixed vocabulary size of 32,768 (Section B EXPERIMENTAL DETAILS; Section 4.1 EXPERIMENTAL DESIGN).
> - Model sharing vs specialization: Shared pre-trained model with per-task fine-tuning (Section B EXPERIMENTAL DETAILS).
> - Role of positional encoding: Central experimental variable with multiple variants and ablations (ABSTRACT; IMPLEMENTATION DETAILS AND DISCUSSIONS; Section 4.1 EXPERIMENTAL DESIGN).

### 14. Final Classification

**Multi-task, single-domain**

The evaluation uses multiple GLUE tasks: "we use nine tasks in GLUE, including CoLA, RTE, MRPC, STS, SST, QNLI, QQP, and MNLI-m/mm." (Section B EXPERIMENTAL DETAILS). The work stays within language/text data, described as "language pretraining" and using "the English Wikipedia corpus and BookCorpus" (ABSTRACT; Section 4.1 EXPERIMENTAL DESIGN), with no cross-domain or multi-modal evaluation claims.
