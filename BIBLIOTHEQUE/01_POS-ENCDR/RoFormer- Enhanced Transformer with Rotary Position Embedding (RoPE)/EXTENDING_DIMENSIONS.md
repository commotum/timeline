## 1. Basic Metadata

- Title: "ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING" (front matter)
- Authors: "Jianlin Su"; "Ahmed Murtadha"; "Yu Lu"; "Bo Wen"; "Shengfeng Pan"; "Yunfeng Liu" (front matter)
- Year: 2023 ("November 9, 2023", front matter)
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper states, "we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information" so that it "encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation" (Abstract).

## 3. Tasks Evaluated

- **Machine translation (English-German)** Task type: Generation. Dataset(s): WMT 2014 English-German. Domain: natural language text (English/German). Evidence: "We first demonstrate the performance of RoFormer on sequence-to-sequence language translation tasks." (Section 4.1) and "We choose the standard WMT 2014 English-German datasetBojar et al. [2014], which consists of approximately 4.5 million sentence pairs." (Section 4.1.1)
- **Masked language modeling pre-training** Task type: Reconstruction. Dataset(s): BookCorpus and Wikipedia Corpus. Domain: natural language text. Evidence: "We use the BookCorpus Zhu et al. [2015] and the Wikipedia Corpus Foundation [2021] from Huggingface Datasets library (Apache License 2.0) for pre-training." (Section 4.2.1) and "We use the masked language-modeling (MLM) loss values of the training process as an evaluation metric." (Section 4.2.1)
- **GLUE downstream tasks (MRPC, SST-2, QNLI, STS-B, QQP, MNLI)** Task type: Classification; Other (semantic textual similarity). Dataset(s): MRPC, SST-2, QNLI, STS-B, QQP, MNLI. Domain: natural language text. Evidence: "We look at several datasets from GLUE, i.e. MRPC Dolan and Brockett [2005], SST-2 Socher et al. [2013], QNLI Rajpurkar et al. [2016], STS-B Al-Natsheh [2017], QQP Chen et al. [2018b] and MNLI Williams et al. [2018]." (Section 4.3.1) and "We use F1-score for MRPC and QQP dataset, spearman correlation for STS-B, and accuracy for the remaining as the evaluation metrics." (Section 4.3.1)
- **Language modeling pre-training (PerFormer with RoPE)** Task type: Generation. Dataset(s): Enwik8. Domain: natural language text (English Wikipedia). Evidence: "We demonstrate its performance with the pre-training task of language modeling." (Section 4.4) and "We carry out tests on the Enwik8 dataset Mahoney [2006], which is from English Wikipedia that includes markup, special characters and text in other languages in addition to English text." (Section 4.4.1)
- **Chinese Similar Case Matching (CAIL2019-SCM)** Task type: Reasoning / relational. Dataset(s): CAIL2019-SCM. Domain: Chinese legal text ("cases published by the Supreme People's Court of China"). Evidence: "We choose Chinese AI and Law 2019 Similar Case Matching (CAIL2019-SCM)Xiao et al. [2019] dataset" (Section 4.5.3), "of cases published by the Supreme People's Court of China." (Section 4.5.3), and "The task is to predict whether the pair (A, B) is closer than (A, C) under a predefined similarity measure." (Section 4.5.3)

## 4. Domain and Modality Scope

- Single domain? No; the evaluation spans multiple text domains and languages, e.g., "WMT 2014 English-German dataset" (Section 4.1.1), "BookCorpus Zhu et al. [2015] and the Wikipedia Corpus Foundation [2021]" (Section 4.2.1), "We carry out tests on the Enwik8 dataset Mahoney [2006], which is from English Wikipedia that includes markup, special characters and text in other languages in addition to English text." (Section 4.4.1), and "additional results on Chinese data." (Section 4.5).
- Multiple domains within the same modality? Yes; all evaluations are natural language text with different datasets and languages, including "English-German" translation (Section 4.1.1) and "Chinese AI and Law 2019 Similar Case Matching (CAIL2019-SCM)" (Section 4.5.3).
- Multiple modalities? Not stated.
- Domain generalization or cross-domain transfer? Not claimed; the closest statement is task generalization: "we fine-tune the weights of our pre-trained RoFormer across various GLUE tasks in order to evaluate its generalization ability on the downstream NLP tasks." (Section 4.3)

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Machine translation (WMT14 En-De) | Not specified | Not specified | Not specified | "We train the baseline model and our RoFormer under the same settings" (Section 4.1.3) and "We carry out some modifications on self-attention layer of the baseline model Vaswani et al. [2017] to enable RoPE to its learning process." (Section 4.1.2) |
| Masked language modeling pre-training | Pre-trained weights are later used for downstream fine-tuning | No (pre-training stage) | Not specified | "We use the BookCorpus Zhu et al. [2015] and the Wikipedia Corpus Foundation [2021] from Huggingface Datasets library (Apache License 2.0) for pre-training." (Section 4.2.1) and "we fine-tune the weights of our pre-trained RoFormer across various GLUE tasks" (Section 4.3) |
| GLUE downstream tasks | Pre-trained RoFormer weights shared as initialization | Yes, per task | Not specified | "we fine-tune the weights of our pre-trained RoFormer across various GLUE tasks" (Section 4.3) |
| Language modeling pre-training (PerFormer with RoPE) | Not specified | Not specified | Not specified | "We demonstrate its performance with the pre-training task of language modeling." (Section 4.4) and "We incorporate RoPE into the 12 layer char-based PerFormer with 768 dimensions and 12 heads" (Section 4.4.1) |
| CAIL2019-SCM | Uses a pre-trained RoFormer model | Not specified | Not specified | "We apply the pre-trained RoFormer model to CAIL2019-SCM with different input lengths." (Section 4.5.4) |

## 6. Input and Representation Constraints

- Maximum sequence length of 512 in English pre-training: "We train both BERT and RoFormer with batch size 64 and maximum sequence length of 512 for 100k steps." (Section 4.2.2)
- Maximum sequence length of 512 in GLUE fine-tuning: "with a maximum sequence length of 512" (Section 4.3.2)
- Fixed maximum sequence length of 1024 in the Performer experiment: "a fixed maximum sequence length of 1024" (Section 4.4.1)
- Variable maximum sequence lengths across Chinese pre-training stages: "The training procedure is divided into various consecutive stages. In each stage, we train the model with a specific combination of maximum sequence length and batch size." (Section 4.5.2)
- Long-document constraint: "long documents whose length exceeds 512 characters." (Section 4.5)
- Tokenization/representation level: "Tokenization level | char                     | word            | char                   | word     |" (Table 3) and "12 layer char-based PerFormer" (Section 4.4.1)
- Embedding dimensionality constraint: "any  $x_i \in \mathbb{R}^d$  where d is even" (Section 3.2.2)
- Fixed patch size / image resolution: Not specified.
- Fixed number of tokens (beyond max sequence length caps): Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length values explicitly stated include "maximum sequence length of 512" (Section 4.2.2) and "a fixed maximum sequence length of 1024" (Section 4.4.1).
- Sequence length is capped and sometimes fixed: "maximum sequence length of 512" (Section 4.2.2), "a fixed maximum sequence length of 1024" (Section 4.4.1), and "changing batch size and maximum input sequence length" across stages (Section 4.5.2).
- Attention type: global self-attention for standard Transformer, since "The original self-attention should compute the inner product of query and key for every pair of tokens, which has a quadratic complexity  $\mathbb{O}(N^2)$ ." (Section 3.3)
- Linear attention for Performer: "Performer Choromanski et al. [2020] introduces an alternative attention mechanism, linear attention, which is designed to avoid quadratic computation cost that scales with input sequence length." (Section 4.4)
- Computational cost management: linear attention is used to avoid quadratic cost (Section 4.4, quote above).

## 8. Positional Encoding (Critical Section)

- Mechanism: RoPE, which "encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation." (Abstract)
- Where applied: in the self-attention block, e.g., "we replace the original sinusoidal position encoding in the self-attention block of the baseline model with our proposed RoPE" (Section 4.2.2) and "We carry out some modifications on self-attention layer of the baseline model Vaswani et al. [2017] to enable RoPE to its learning process." (Section 4.1.2)
- Fixed vs modified per task / comparisons: RoPE replaces other positional encodings in different settings, e.g., "we replace the original sinusoidal position encoding of BERT with our RoPE" (Section 4.2) and "replacing the absolute position embedding with our proposed RoPE" (Section 4.5.1); Table 3 contrasts "Position embedding | abs.                     | abs.            | rel.                   | RoPE     |" (Table 3).

## 9. Positional Encoding as a Variable

- Core research variable: Yes; "we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE)" (Abstract).
- Multiple positional encodings compared: Yes; e.g., "we replace the original sinusoidal position encoding of BERT with our RoPE" (Section 4.2) and Table 3 lists "Position embedding | abs.                     | abs.            | rel.                   | RoPE     |" (Table 3).
- Claim that PE choice is not critical or secondary: Not claimed.

## 10. Evidence of Constraint Masking

- Model size(s): "12 layer char-based PerFormer with 768 dimensions and 12 heads" (Section 4.4.1).
- Dataset size(s): "approximately 4.5 million sentence pairs" for WMT 2014 En-De (Section 4.1.1); "approximately 34GB of data collected from Chinese Wikipedia, news and forums" (Section 4.5.2); "CAIL2019-SCM contains 8964 triplets" (Section 4.5.3).
- Performance gains attribution: "Experimental results on various long text classification benchmark datasets show that the enhanced transformer with rotary position embedding, namely RoFormer, can give better performance compared to baseline alternatives and thus demonstrates the efficacy of the proposed RoPE." (Abstract) and "We claim that this is the attribute to the excellent generalizability of the proposed RoPE." (Section 4.5.2)
- Training tricks/settings noted: "learning rate is increased linearly from 1e - 7 to 5e - 4 and then decayed proportionally to the inverse square root of the step number" and "Label smoothing with 0.1 is also adopted." (Section 4.1.2)
- Scaling model size/data as the primary driver: Not claimed.

## 11. Architectural Workarounds

- Linear attention to reduce complexity: "Performer Choromanski et al. [2020] introduces an alternative attention mechanism, linear attention, which is designed to avoid quadratic computation cost that scales with input sequence length." (Section 4.4)
- RoPE for long-sequence handling: "RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances" (Abstract).
- Self-attention modifications to integrate RoPE: "We carry out some modifications on self-attention layer of the baseline model Vaswani et al. [2017] to enable RoPE to its learning process." (Section 4.1.2)

## 12. Explicit Limitations and Non-Claims

- Limitations: "there lacks of thorough explanations on why it converges faster than baseline models that incorporates other position encoding strategies." (Section 4.5.5)
- Limitations: "we have not come up with a faithful explanation." (Section 4.5.5)
- Resource limitation: "Our proposed RoFormer is built upon the Transformer-based infrastructure, which requires hardware resources for pre-training purpose." (Section 4.5.5)
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Text-only NLP across English and Chinese datasets, e.g., "WMT 2014 English-German dataset" (Section 4.1.1) and "additional results on Chinese data." (Section 4.5).
> – Task structure: Multiple distinct NLP tasks, as stated: "We evaluate the proposed RoFormer on various NLP tasks as follows." (Section 4)
> – Representation rigidity: Explicit sequence-length caps such as "maximum sequence length of 512" (Section 4.2.2) and a "fixed maximum sequence length of 1024" (Section 4.4.1), plus an even-dimensional constraint ("d is even", Section 3.2.2).
> – Model sharing vs specialization: A shared pre-trained backbone is fine-tuned for downstream tasks ("we fine-tune the weights of our pre-trained RoFormer across various GLUE tasks", Section 4.3), while translation is trained as its own task ("We train the baseline model and our RoFormer under the same settings", Section 4.1.3).
> – Role of positional encoding: Central variable, explicitly swapped in for alternatives ("replace the original sinusoidal position encoding of BERT with our RoPE", Section 4.2; "replacing the absolute position embedding with our proposed RoPE", Section 4.5.1).

### 14. Final Classification

**Multi-task, multi-domain (constrained).** The paper explicitly states, "We evaluate the proposed RoFormer on various NLP tasks as follows" and then covers translation, pre-training language modeling, GLUE tasks, and Chinese similar-case matching (Section 4, Sections 4.1-4.5). It also spans multiple text domains and languages, including "WMT 2014 English-German dataset" (Section 4.1.1) and "additional results on Chinese data" with CAIL2019-SCM (Section 4.5), while remaining within the text modality, indicating a constrained multi-task, multi-domain setup.
