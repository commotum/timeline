# ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING (2023)
Source: RoFormer- Enhanced Transformer with Rotary Position Embedding (RoPE).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Machine translation (English-to-German generation) | Source-language text token sequence | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Target-language text token sequence | 1D (t) (inferred) | Not specified in the paper. |
| Language modeling pre-training (MLM / token prediction) | Text token sequences | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Token predictions for sequence positions | 1D (t) (inferred) | Capped (inferred) |
| Downstream natural language understanding (GLUE classification/similarity scoring) | Sentence or sentence-pair token sequences | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Class label or similarity score | 0D (inferred) | Fixed (inferred) |
| Semantic text matching (CAIL2019-SCM similar-case ranking) | Triplets of case-description token sequences (A, B, C) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Binary decision of whether (A, B) is closer than (A, C) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates RoFormer on text-only NLP tasks spanning sequence generation/prediction (machine translation and language modeling) and decision tasks (GLUE classification/similarity and legal semantic matching). All supported tasks operate over token sequences, so the defensible input address space is 1D (t), while outputs are either token sequences (1D) or single decisions/scores (0D). Dynamics are explicitly capped where maximum sequence lengths are given (e.g., 512 and 1024), while translation length limits are not explicitly specified. From the described self-attention setup, attention is Static and state is Direct across reported tasks (inferred).

## Evidence
### Task: Machine translation (English-to-German generation)
- "We first demonstrate the performance of RoFormer on sequence-to-sequence language translation tasks." (Section 4.1)
- "We choose the standard WMT 2014 English-German datasetBojar et al. [2014], which consists of approximately 4.5 million sentence pairs." (Section 4.1.1)
- Inference: Input/output were labeled as 1D token sequences from "a joint source and target byte pair encoding(BPE)" and sequence-to-sequence translation wording; Attention Dynamic was labeled Static and State Dynamic Direct from the paper’s standard self-attention formulation where all tokens in the provided sequence are attended within the given input slice (Sections 2.1, 4.1.2). In/Out Dynamics are "Not specified in the paper." because no explicit translation max-length interface is given.

### Task: Language modeling pre-training (MLM / token prediction)
- "We use the BookCorpus Zhu et al. [2015] and the Wikipedia Corpus Foundation [2021] from Huggingface Datasets library (Apache License 2.0) for pre-training." (Section 4.2.1)
- "We use the masked language-modeling (MLM) loss values of the training process as an evaluation metric." (Section 4.2.1)
- "We train both BERT and RoFormer with batch size 64 and maximum sequence length of 512 for 100k steps." (Section 4.2.2)
- Inference: Input/output dimensions were labeled 1D (t) because MLM operates on ordered token sequences and predicts sequence-position tokens; In/Out Dynamics were labeled Capped from explicit "maximum sequence length" constraints (Sections 4.2.2 and 4.4.1 with fixed maximum 1024); Attention Dynamic was labeled Static and State Dynamic Direct from the described self-attention computation over the provided sequence (Sections 2.1, 3.3).

### Task: Downstream natural language understanding (GLUE classification/similarity scoring)
- "Consistent with the previous experiments, we fine-tune the weights of our pre-trained RoFormer across various GLUE tasks in order to evaluate its generalization ability on the downstream NLP tasks." (Section 4.3)
- "We look at several datasets from GLUE, i.e. MRPC Dolan and Brockett [2005], SST-2 Socher et al. [2013], QNLI Rajpurkar et al. [2016], STS-B Al-Natsheh [2017], QQP Chen et al. [2018b] and MNLI Williams et al. [2018]. We use F1-score for MRPC and QQP dataset, spearman correlation for STS-B, and accuracy for the remaining as the evaluation metrics." (Section 4.3.1)
- "We use Huggingface Transformers library (Apache License 2.0)Wolf et al. [2020] to fine-tune each of the aforementioned downstream tasks for 3 epochs, with a maximum sequence length of 512, batch size of 32 and learning rates 2,3,4,5e-5." (Section 4.3.2)
- Inference: Input was labeled 1D (t) token sequences from sentence/sentence-pair GLUE tasks; In Dynamics was labeled Capped from explicit maximum sequence length 512; Output was labeled 0D with Fixed dynamics because each example yields one metric-targeted decision/score (classification label or similarity score); Attention Dynamic was labeled Static and State Dynamic Direct from the same self-attention runtime description used throughout the paper (Sections 2.1, 3.3).

### Task: Semantic text matching (CAIL2019-SCM similar-case ranking)
- "We choose Chinese AI and Law 2019 Similar Case Matching (CAIL2019-SCM)Xiao et al. [2019] dataset to illustrate the ability of RoFormer in dealing with long texts, i.e., semantic text matching." (Section 4.5.3)
- "The input triplet, denoted as (A, B and C), are fact descriptions of three cases. The task is to predict whether the pair (A, B) is closer than (A, C) under a predefined similarity measure." (Section 4.5.3)
- "With short text cut-offs, i.e., 512, the result from RoFormer is comparable to WoBERT and is slightly better than the BERT implementation. However, when increasing the maximum input text length to 1024, RoFormer outperforms WoBERT by an absolute improvement of 1.5%." (Section 4.5.4)
- Inference: Input dimension was labeled 1D (t) because each case description is a token sequence; In Dynamics was labeled Capped from explicit cut-off/maximum lengths (512/1024); Output was labeled 0D with Fixed dynamics because the task is a single decision for each triplet; Attention Dynamic was labeled Static and State Dynamic Direct from the same transformer self-attention framing in the paper (Sections 2.1, 3.3).
