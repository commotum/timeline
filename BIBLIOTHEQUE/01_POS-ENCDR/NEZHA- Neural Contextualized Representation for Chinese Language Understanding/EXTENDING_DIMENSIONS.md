## 1. Basic Metadata

- Title: "NEZHA: NEURAL CONTEXTUALIZED REPRESENTATION FOR CHINESE LANGUAGE UNDERSTANDING" (front matter)
- Authors: "Junqiu Wei, Xiaozhe Ren, Xiaoguang Li, Wenyong Huang, Yi Liao, Yasheng Wang, Jiashu Lin*, Xin Jiang, Xiao Chen, Qun Liu" (front matter)
- Year: "November 22, 2021" (front matter)
- Venue: "TECHNICAL REPORT" (front matter)

## 2. One-Sentence Contribution Summary

The contribution is to "present our practice of pre-training language models named NEZHA (NEural contextualiZed representation for CHinese lAnguage understanding) on Chinese corpora and finetuning for the Chinese NLU tasks" and to "assess the effectiveness of training factors including positional encoding scheme, masking strategy, sources of training corpora, length of training sequences" (ABSTRACT; Introduction).

## 3. Tasks Evaluated

Task name: CMRC (Chinese Machine Reading Comprehension 2018)  
Task type: Other (span extraction / machine reading comprehension)  
Dataset(s) used: "CMRC (Chinese Machine Reading Comprehension 2018)" (3.2 Experimental Results)  
Domain: "Wikipedia" (Table 3)  
Evidence: "CMRC (Chinese Machine Reading Comprehension 2018) [16]: A machine reading comprehension task that returns an answer span in a given passage for a given question." (3.2 Experimental Results); "| CMRC      | 16/72        | 384 | 3e-5 | 2      | 10K    | 3.2K | -     | Wikipedia |" (Table 3)

Task name: XNLI (Cross-lingual Natural Language Inference)  
Task type: Classification  
Dataset(s) used: "XNLI (Cross-lingual Natural Language Inference)" (3.2 Experimental Results)  
Domain: "General" (Table 3)  
Evidence: "XNLI (Cross-lingual Natural Language Inference) [17]: The Chinese portion of XNLI, which is a version of MultiNLI where the dev and test sets have been translated (by humans) into 15 languages. XNLI is a natural language inference task. The goal of this task is to predict if the second sentence is a contradiction, entailment or neutral to the first sentence." (3.2 Experimental Results); "| XNLI      | 64/32        | 128 | 3e-5 | 3      | 392K   | 2.5K | 2.5K  | General   |" (Table 3)

Task name: LCQMC (Large-scale Chinese Question Matching Corpus)  
Task type: Classification  
Dataset(s) used: "LCQMC (Large-scale Chinese Question Matching Corpus)" (3.2 Experimental Results)  
Domain: "QA" (Table 3)  
Evidence: "LCQMC (Large-scale Chinese Question Matching Corpus) [18]: A sentence pair matching task. Given a pair of sentences, the task is to determine if the two sentences are semantically equivalent or not." (3.2 Experimental Results); "| LCQMC     | 64/32        | 128 | 3e-5 | 5      | 240K   | 8.8K | 12.5K | QA        |" (Table 3)

Task name: PD-NER (People's Daily Named Entity Recognition)  
Task type: Other (sequence labeling / NER)  
Dataset(s) used: "PD-NER (People's Daily Named Entity Recognition)" (3.2 Experimental Results)  
Domain: "News" (Table 3)  
Evidence: "**PD-NER** (People's Daily Named Entity Recognition) <sup>9</sup>: A sequence labeling task that identifies the named entities from text. The corpus is from *People's Daily*, a Chinese News Media." (3.2 Experimental Results); "| PD-NER    | 64/16        | 256 | 3e-5 | 5      | 51K    | 4.6K | 68    | News      |" (Table 3)

Task name: ChnSenti (Chinese Sentiment Classification)  
Task type: Classification  
Dataset(s) used: "ChnSenti (Chinese Sentiment Classification)" (3.2 Experimental Results)  
Domain: "General" (Table 3)  
Evidence: "**ChnSenti** (Chinese Sentiment Classification) <sup>10</sup>: A binary classification task which predicts if the sentiment of a given sentence is positive or negative." (3.2 Experimental Results); "| ChnSenti  | 64/16        | 256 | 3e-5 | 10     | 9.6K   | 1.2K | 1.2K  | General   |" (Table 3)

## 4. Domain and Modality Scope

- Evaluation domains: Multiple domains within the same modality (text), e.g., "Wikipedia," "General," "QA," and "News" (Table 3).
- Modality: Chinese text NLU tasks, described as "Chinese NLU tasks" and "Chinese text" (ABSTRACT; 3.1 Experimental Setting).
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| CMRC | Yes (shared pretrained NEZHA) | Yes | Not specified. | "pre-training language models named NEZHA (NEural contextualiZed representation for CHinese lAnguage understanding) on Chinese corpora and finetuning for the Chinese NLU tasks" (ABSTRACT); "We test the performances of the pre-trained models by fine-tuning on a variety of natural language understanding (NLU) tasks" (3.2 Experimental Results) |
| XNLI | Yes (shared pretrained NEZHA) | Yes | Not specified. | "pre-training language models named NEZHA (NEural contextualiZed representation for CHinese lAnguage understanding) on Chinese corpora and finetuning for the Chinese NLU tasks" (ABSTRACT); "We test the performances of the pre-trained models by fine-tuning on a variety of natural language understanding (NLU) tasks" (3.2 Experimental Results) |
| LCQMC | Yes (shared pretrained NEZHA) | Yes | Not specified. | "pre-training language models named NEZHA (NEural contextualiZed representation for CHinese lAnguage understanding) on Chinese corpora and finetuning for the Chinese NLU tasks" (ABSTRACT); "We test the performances of the pre-trained models by fine-tuning on a variety of natural language understanding (NLU) tasks" (3.2 Experimental Results) |
| PD-NER | Yes (shared pretrained NEZHA) | Yes | Not specified. | "pre-training language models named NEZHA (NEural contextualiZed representation for CHinese lAnguage understanding) on Chinese corpora and finetuning for the Chinese NLU tasks" (ABSTRACT); "We test the performances of the pre-trained models by fine-tuning on a variety of natural language understanding (NLU) tasks" (3.2 Experimental Results) |
| ChnSenti | Yes (shared pretrained NEZHA) | Yes | Not specified. | "pre-training language models named NEZHA (NEural contextualiZed representation for CHinese lAnguage understanding) on Chinese corpora and finetuning for the Chinese NLU tasks" (ABSTRACT); "We test the performances of the pre-trained models by fine-tuning on a variety of natural language understanding (NLU) tasks" (3.2 Experimental Results) |

## 6. Input and Representation Constraints

- Input structure (pre-training): "Each sample in the training data of BERT is a pair of sentences." (2.1 Preliminaries: BERT Model & Positional Encoding)
- Masking assumptions (pre-training): "In each sample, 12% tokens are masked and 1.5% tokens are randomly replaced by another token in the vocabulary." (2.1 Preliminaries: BERT Model & Positional Encoding)
- Fixed number of tokens / sequence length: "Table 3: Hyperparameters used in finetuning downstream tasks. (SL: sequence length; LR stands: learning rate.)" (Table 3); examples include "| CMRC      | 16/72        | 384 | 3e-5 | 2      | 10K    | 3.2K | -     | Wikipedia |" and "| XNLI      | 64/32        | 128 | 3e-5 | 3      | 392K   | 2.5K | 2.5K  | General   |" (Table 3); ablation notes "when trained with a maximum of 128 tokens" and uses "| News, FRPE, SL:512                  | 67.79 | 86.60 | 80.57 | 79.52 | 90.06 | 86.73 | 97.04  | 97.62 | 95.09    | 95.08 |" (3.3 Ablation Study; Table 5).
- Fixed input resolution / patch size: Not specified.
- Fixed dimensionality (e.g., strictly 2D): Not specified.
- Padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: "| News, FRPE, SL:512                  | 67.79 | 86.60 | 80.57 | 79.52 | 90.06 | 86.73 | 97.04  | 97.62 | 95.09    | 95.08 |" (Table 5); per-task SL includes "384" for CMRC in "| CMRC      | 16/72        | 384 | 3e-5 | 2      | 10K    | 3.2K | -     | Wikipedia |" (Table 3).
- Fixed or variable length: "SL: sequence length" is specified as a hyperparameter per task; variability is not described (Table 3).
- Attention type: Global self-attention over all tokens is implied by "z_i = \sum_{i=1}^n \alpha_{ij}(x_j W^V)" with a sum over all positions (2.1 Preliminaries: BERT Model & Positional Encoding).
- Mechanisms to manage computational cost (windowing, pooling, token pruning): Not specified.

## 8. Positional Encoding (Critical Section)

- Mechanism: Functional relative positional encoding with sinusoidal functions of relative position, fixed (non-trainable). Evidence: "we employ functional relative positional encoding, where the computation of the outputs and attention scores involves sinusoidal functions of their relative position" (2.2 Functional Relative Positional Encoding); "encodes the relative positions in self-attention by pre-defined functions without any trainable parameter" (Introduction); "a_{ij}^V and a_{ij}^R are both derived from sinusoidal functions and fixed during the model training" (2.2 Functional Relative Positional Encoding).
- Where it is applied: in self-attention scores and values, using relative-position vectors in attention computation. Evidence: "the computation of the attention scores involves a parametric embedding regarding the relative distance between the two positions" (2.1 Preliminaries: BERT Model & Positional Encoding); "we employ functional relative positional encoding" in which "the computation of the outputs and attention scores" uses relative position (2.2 Functional Relative Positional Encoding).
- Fixed vs modified per task / ablated: Positional encoding is compared across alternatives: "Positional Encoding: the effectiveness of the functional relative positional encoding (FRPE) employed in our work compared with the parametric absolute positional encoding (PAPE) and parametric relative positional encoding (PRPE)" (3.3 Ablation Study).

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption: Treated as a research variable, explicitly studied in ablation: "Positional Encoding: the effectiveness of the functional relative positional encoding (FRPE) employed in our work compared with the parametric absolute positional encoding (PAPE) and parametric relative positional encoding (PRPE)" (3.3 Ablation Study).
- Multiple positional encodings compared: Yes, FRPE vs PAPE vs PRPE (3.3 Ablation Study).
- Claims PE choice is not critical: Not claimed; instead, "functional relative positional encoding shows a notable advantage compared with other positional encoding methods" (3.3 Ablation Study).

## 10. Evidence of Constraint Masking

- Model sizes: "| NEZHA <sub>BASE</sub><br>NEZHA <sub>LARGE</sub>                     | Wikipedia+Baike+Nev  | vs 10,536M | 21,128          | GeLU                | 768/12<br>1024/24      | 12<br>16 |" (Table 1).
- Dataset sizes: "Chinese Wikipedia <sup>3</sup>. Chinese Wikipedia is a Chinese-language encyclopedia containing 1,067,552 articles. We downloaded the latest Chinese Wikipedia dump and cleaned the raw data with the tool named WikiExtractor<sup>4</sup>. The cleaned corpus contains both simplified and traditional Chinese and has roughly 202M tokens." (3.1 Experimental Setting); "Baidu Baike <sup>5</sup>. We crawled webpages from the Baidu Baike, which is a Chinese-language, collaborative, web-based encyclopedia owned and produced by the Chinese search engine Baidu. As of August 2018, Baidu Baike has more than 15.4 million articles. The cleaned corpus contains 4,734M tokens." (3.1 Experimental Setting); "Chinese News. We crawled Chinese News corpus from multiple news websites (e.g., Sina News). The cleaned corpus contains 5,600M tokens." (3.1 Experimental Setting).
- Reported performance gains are attributed to architectural/training choices rather than scaling alone: "the unique technique in our models is the functional relative position encoding" (3.2 Experimental Results); ablation reports "functional relative positional encoding shows a notable advantage compared with other positional encoding methods" (3.3 Ablation Study).
- Scaling model size or data as primary driver: Not explicitly stated.

## 11. Architectural Workarounds

- Functional Relative Positional Encoding (FRPE): "encodes the relative positions in self-attention by pre-defined functions without any trainable parameter" and uses fixed sinusoidal functions to help extrapolate to longer sequences (Introduction; 2.2 Functional Relative Positional Encoding: "We choose the fixed sinusoidal functions mainly because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.")
- Mixed precision training to manage scale: "can speed up the training by 2-3 times and also reduce the space consumption of the model, as a result of which, a larger batch size could be utilized" (2.4 Mixed Precision Training).
- LAMB optimizer for large-batch training: "designed for the large batch-size synchronous distributed training" and "speeds up the training of BERT by using a very large batch size" (2.5 LAMB Optimizer).

## 12. Explicit Limitations and Non-Claims

- Future work: "In the future, we plan to continue the work on improving NEZHA on Chinese and other languages and extend the applications of NEZHA to more scenarios." (4 Conclusion)
- Explicit limitations or non-claims (e.g., open-world learning, unrestrained multi-task learning, meta-learning): Not specified.

### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: evaluated on "Chinese NLU tasks" across domains "Wikipedia," "General," "QA," and "News" (ABSTRACT; Table 3).
- Task structure: multiple supervised NLU tasks including MRC, NLI, sentence matching, NER, sentiment (3.2 Experimental Results).
- Representation rigidity: fixed sequence-length hyperparameters ("SL: sequence length" with values like 128/256/384, and ablation "SL:512") (Table 3; Table 5).
- Model sharing vs specialization: shared NEZHA pretraining with per-task fine-tuning (ABSTRACT; 3.2 Experimental Results).
- Role of positional encoding: FRPE is central and compared against PAPE/PRPE (Introduction; 3.3 Ablation Study).

### 14. Final Classification

**Multi-task, multi-domain (constrained).** The paper evaluates "a variety of natural language understanding (NLU) tasks" (3.2 Experimental Results) across multiple text domains listed as "Wikipedia," "General," "QA," and "News" (Table 3). All evaluations remain within the single modality of Chinese text and rely on pretraining plus fine-tuning rather than unrestrained multi-task learning (ABSTRACT; 3.2 Experimental Results).
