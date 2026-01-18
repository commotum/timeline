## 1. Basic Metadata

- Title: "TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION"
- Authors: "Ofir Press<sup>1,2</sup> Noah A. Smith<sup>1,3</sup> Mike Lewis<sup>2</sup>"
- Year: Year not specified.
- Venue: Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper proposes ALiBi, a position-representation change that biases attention scores to enable efficient length extrapolation in transformer language models.

---

## 3. Tasks Evaluated

### Task 1: Language modeling (next-token prediction) on WikiText-103

- Task type: Generation
- Dataset(s) used: WikiText-103
- Domain: English Wikipedia text (natural language)
- Evidence (quotes):
  - "A transformer LM receives a list of tokens and outputs a probability distribution representing its prediction for the next token." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)
  - "We first test the extrapolation abilities of various position methods on the WikiText-103 corpus (Merity et al., 2016) using the transformer language model of Baevski & Auli (2018)." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)
  - "The training set is about 103 million tokens from English Wikipedia (half a gigabyte)." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)

### Task 2: Language modeling (next-token prediction) on Toronto BooksCorpus

- Task type: Generation
- Dataset(s) used: Toronto BooksCorpus
- Domain: Books text
- Evidence (quotes):
  - "A transformer LM receives a list of tokens and outputs a probability distribution representing its prediction for the next token." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)
  - "We emphasize that our set of slopes was chosen by running experiments on the WikiText-103 corpus, and here we apply that set of slopes to a model trained on a very different text domain." (Appendix A.3 RESULTS ON THE TORONTO BOOK CORPUS)
  - "Specifically, we use the Toronto BooksCorpus (Zhu et al., 2015), which has been used to train BERT (Devlin et al., 2019) (in conjuction with the English Wikipedia)." (Appendix A.3 RESULTS ON THE TORONTO BOOK CORPUS)
  - "The corpus is about 700M tokens (2.9 GB)." (Appendix A.3 RESULTS ON THE TORONTO BOOK CORPUS)

### Task 3: Language modeling (next-token prediction) on CC100+RoBERTa corpus

- Task type: Generation
- Dataset(s) used: CC100+RoBERTa corpus (RoBERTa corpus + English CC-100)
- Domain: English text corpora
- Evidence (quotes):
  - "A transformer LM receives a list of tokens and outputs a probability distribution representing its prediction for the next token." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)
  - "The dataset we choose is a combination of the datasets used to train the RoBERTa (Liu et al., 2019) implementation of BERT (Devlin et al., 2019) and the English part of the CC-100 corpus introduced in Conneau et al. (2020), for a total of 461 GB." (Section 4.2 RESULTS ON THE CC100+ROBERTA CORPUS)
  - "The validation set contains 649K tokens." (Section 4.2 RESULTS ON THE CC100+ROBERTA CORPUS)

---

## 4. Domain and Modality Scope

- Evaluation domain scope: Multiple domains within the same modality (text). Evidence includes "The training set is about 103 million tokens from English Wikipedia (half a gigabyte)." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP), "Specifically, we use the Toronto BooksCorpus (Zhu et al., 2015), which has been used to train BERT (Devlin et al., 2019) (in conjuction with the English Wikipedia)." (Appendix A.3 RESULTS ON THE TORONTO BOOK CORPUS), and "The dataset we choose is a combination of the datasets used to train the RoBERTa (Liu et al., 2019) implementation of BERT (Devlin et al., 2019) and the English part of the CC-100 corpus introduced in Conneau et al. (2020), for a total of 461 GB." (Section 4.2 RESULTS ON THE CC100+ROBERTA CORPUS).
- Multiple modalities? Not stated; the evaluation data described are text corpora (quotes above).
- Domain generalization / cross-domain transfer: Claimed. "We emphasize that our set of slopes was chosen by running experiments on the WikiText-103 corpus, and here we apply that set of slopes to a model trained on a very different text domain." (Appendix A.3 RESULTS ON THE TORONTO BOOK CORPUS) and "This result establishes the generality of ALiBi and the particular set of slopes we found and suggests that they may be used on different text domains without further hyperparameter tuning." (Appendix A.3 RESULTS ON THE TORONTO BOOK CORPUS)

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Language modeling on WikiText-103 | Not specified. | Not specified. | Not specified. | "We first test the extrapolation abilities of various position methods on the WikiText-103 corpus (Merity et al., 2016) using the transformer language model of Baevski & Auli (2018)." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP) |
| Language modeling on Toronto BooksCorpus | Not specified. | Not specified. | Not specified. | "We emphasize that our set of slopes was chosen by running experiments on the WikiText-103 corpus, and here we apply that set of slopes to a model trained on a very different text domain." (Appendix A.3 RESULTS ON THE TORONTO BOOK CORPUS) |
| Language modeling on CC100+RoBERTa corpus | Not specified. | Not specified. | Not specified. | "We train our models for one epoch, which is 50k updates on 128 V100 GPUs." (Section 4.2 RESULTS ON THE CC100+ROBERTA CORPUS) |

---

## 6. Input and Representation Constraints

- Fixed number of tokens per training subsequence: "Let L be the length of each input subsequence during training; it includes L predictions, which on average have access to  $\frac{L+1}{2}$  tokens of (left) context." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)
- Handling longer sequences by segmentation: "To train on or evaluate a sequence longer than L tokens, it is typical to segment the sequence into L-length subsequences and train on or evaluate them independently." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)
- Variable-length evaluation regime: "To explore a model's extrapolation abilities, we are interested in cases where sequences of length  $L_{valid} > L$  are considered at evaluation time." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)
- Variable-length-capable layers: "the functions that define a transformer layer are agnostic to input length; they map from some arbitrary, unfixed number of input vectors to the same number of output vectors." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)
- Fixed patch size: Not specified.
- Fixed input resolution: Not specified.
- Fixed dimensionality (e.g., strictly 2D): Not specified.
- Padding or resizing requirements: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length tested: "we then run inference with it on the validation set on L+k tokens, with k ranging from 0 to 15,000." (Section 2.2 Measuring Extrapolation) and "ALiBi maintains strong performance even on sequences of length 10,000." (Section 1 Introduction)
- Fixed or variable sequence length: "Let L be the length of each input subsequence during training; it includes L predictions, which on average have access to  $\frac{L+1}{2}$  tokens of (left) context." and "To explore a model's extrapolation abilities, we are interested in cases where sequences of length  $L_{valid} > L$  are considered at evaluation time." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)
- Attention type: Global causal attention. "using a \"causal mask\" that ensures each position's prediction is influenced only by tokens to its left." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)
- Mechanisms to manage computational cost: "To train on or evaluate a sequence longer than L tokens, it is typical to segment the sequence into L-length subsequences and train on or evaluate them independently." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)

---

## 8. Positional Encoding (Critical Section)

- ALiBi mechanism (bias-based relative position): "ALiBi does not add positional embeddings to word embeddings; instead, it biases query-key attention scores with a penalty that is proportional to their distance." (Abstract)
- Application point and learnability: "When using ALiBi, we do not add position embeddings at any point in the network. The only modification we apply is after the query-key dot product, where we add a static, non-learned bias:" (Section 3 ATTENTION WITH LINEAR BIASES (ALIBI)) and "where scalar m is a head-specific slope fixed before training." (Section 3 ATTENTION WITH LINEAR BIASES (ALIBI))
- Layer-wise application: "Since ALiBi is a relative position method, we add position information at every layer to the keys and queries but not to the values, as is done in the T5 bias and rotary methods." (Section 3 ATTENTION WITH LINEAR BIASES (ALIBI))
- Baseline positional encodings compared: "are constant, non-learned vectors that are added to token embeddings on input to the first layer of the transformer." (Section 2.2 Measuring Extrapolation); "they multiply the keys and queries of every attention layer by sinusoidal embeddings." (Section 2.2 Measuring Extrapolation); "we add a learned, shared bias to each query-key score that is dependent on just the distance between the query and key." (Section 2.2 Measuring Extrapolation)
- Fixed vs modified per task: "We emphasize that our set of slopes was chosen by running experiments on the WikiText-103 corpus, and here we apply that set of slopes to a model trained on a very different text domain." (Appendix A.3 RESULTS ON THE TORONTO BOOK CORPUS) and "Throughout the entire process of developing this method, we ran only one set of experiments on this domain using the previously selected set of slopes." (Appendix A.3 RESULTS ON THE TORONTO BOOK CORPUS)

---

## 9. Positional Encoding as a Variable

- Core research variable: "We demonstrate that this failure to extrapolate is caused by the position embedding method." (Section 1 Introduction) and "we conclude that extrapolation ability depends heavily on the position embedding." (Section 2 CURRENT APPROACHES DO NOT EXTRAPOLATE EFFICIENTLY)
- Multiple positional encodings compared: "Figure 2: A comparison of batched training, inference speed and memory use of the sinusoidal, rotary, T5 bias, and our ALiBi position methods." (Section 2 CURRENT APPROACHES DO NOT EXTRAPOLATE EFFICIENTLY)
- PE choice as secondary/not critical: Not claimed; the paper explicitly ties extrapolation to the position method (quotes above).

---

## 10. Evidence of Constraint Masking

- Model sizes: "These models have 1.3B parameters." (Section 4.2 RESULTS ON THE CC100+ROBERTA CORPUS) and "The model has 16 transformer layers of dimension 1024, with 8 heads, and a feedforward inner dimension of 4096." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)
- Dataset sizes: "The training set is about 103 million tokens from English Wikipedia (half a gigabyte)." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP); "The corpus is about 700M tokens (2.9 GB)." (Appendix A.3 RESULTS ON THE TORONTO BOOK CORPUS); "for a total of 461 GB." (Section 4.2 RESULTS ON THE CC100+ROBERTA CORPUS)
- Training compute: "We train our models for one epoch, which is 50k updates on 128 V100 GPUs." (Section 4.2 RESULTS ON THE CC100+ROBERTA CORPUS)
- Attribution of gains: "We first show that extrapolation can be enabled by simply changing the position representation method, though we find that current methods do not allow for *efficient* extrapolation." (Abstract) and "We demonstrate that this failure to extrapolate is caused by the position embedding method." (Section 1 Introduction). A concrete comparison ties gains to ALiBi rather than scale: "We show that this method trains a 1.3 billion parameter model on input sequences of length 1024 that extrapolates to input sequences of length 2048, achieving the same perplexity as a sinusoidal position embedding model trained on inputs of length 2048 but training 11% faster and using 11% less memory." (Abstract)

---

## 11. Architectural Workarounds

- Attention bias instead of positional embeddings: "ALiBi does not add positional embeddings to word embeddings; instead, it biases query-key attention scores with a penalty that is proportional to their distance." (Abstract)
- Implementation via mask modification: "We implement it by modifying the mask matrix by adding the linear biases to it" (Section 3 ATTENTION WITH LINEAR BIASES (ALIBI))
- Causal masking (autoregressive constraint): "using a \"causal mask\" that ensures each position's prediction is influenced only by tokens to its left." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)
- Sequence segmentation for long inputs: "To train on or evaluate a sequence longer than L tokens, it is typical to segment the sequence into L-length subsequences and train on or evaluate them independently." (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP)

---

## 12. Explicit Limitations and Non-Claims

- "Though performance peaks at around two times the number of tokens that the model was trained on, ALiBi maintains strong performance even on sequences of length 10,000." (Section 1 Introduction)
- "Our analysis reveals that when  $L_{valid} > L$ , ALiBi might not be using contexts longer than the ones it was trained on." (Section B.2 EXTRAPOLATION REDUCES THE EARLY TOKEN CURSE)
- "We hypothesize that future work building on ALiBi might achieve further gains by more efficiently exploiting longer histories." (Section B ANALYSIS)
- "To keep our experiments as straightforward as possible, however, we do not add layers to our models." (Section 4.2 RESULTS ON THE CC100+ROBERTA CORPUS)

---

## 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: multiple text corpora within a single modality (language).
> - Task structure: single-task language modeling / next-token prediction with perplexity evaluation.
> - Representation rigidity: fixed-length training subsequences L; longer inputs handled via L_valid and segmentation.
> - Model sharing vs specialization: dataset-specific experiments are described; no joint multi-task training is stated.
> - Role of positional encoding: central experimental variable (ALiBi vs sinusoidal/rotary/T5 bias).

---

## 14. Final Classification

**Single-task, single-domain**

The paper evaluates a single task: a transformer language model that "outputs a probability distribution representing its prediction for the next token" on text corpora such as WikiText-103, Toronto BooksCorpus, and the CC100+RoBERTa mixture (Section 2.1 BACKGROUND AND EXPERIMENTAL SETUP; Appendix A.3 RESULTS ON THE TORONTO BOOK CORPUS; Section 4.2 RESULTS ON THE CC100+ROBERTA CORPUS). All evaluations are within the text modality, and no joint multi-task training is stated.
