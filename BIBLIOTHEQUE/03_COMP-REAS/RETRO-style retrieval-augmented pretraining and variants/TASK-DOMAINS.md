# Improving language models by retrieving from trillions of tokens (Year not specified in the paper)
Source: RETRO-style retrieval-augmented pretraining and variants.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Autoregressive language modeling / text generation | Text token sequences and retrieved token chunks | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Next-token likelihoods/logits and generated text tokens | 1D (t) (inferred) | Capped (inferred) |
| Open-domain question answering | Question tokens plus top-20 DPR retrieved passages and titles | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Answer text tokens | 1D (t) (inferred) | Capped (inferred) |

## Summary
Retro is used for autoregressive language modeling and is later fine-tuned for question answering, both over text-token sequences. The justified input/output dimensionality is 1D (t) across tasks, with capped dynamics from explicit limits like sequence/chunk sizes and top-20 retrieved passages. Attention is dynamic for language modeling because nearest neighbors are retrieved at runtime, but static for the QA setup because DPR passages are provided as a fixed set. State is constructed in both cases because retrieval neighbors are encoded and integrated through cross-attention before token prediction.

## Evidence
### Task: Autoregressive language modeling / text generation
- "Language modelling (LM) is an unsupervised task that consists of modelling the probability of text" (Section 1. Introduction)
- "We introduce Retro, a retrieval-enhanced autoregressive language model (§2.2)." (Section 1. Introduction)
- "Token likelihoods are provided by a model, parameterized by  $\theta$ , that takes as input both previous tokens and their retrieved neighbours." (Section 2.2. Retrieval-enhanced autoregressive token models)
- "**Output:**  $O \in \mathbb{R}^{n \times |\mathbb{V}|}$ : the output logits" (Section 2.4. Retro model architecture, Algorithm 1)
- Inference: `1D (t)` is inferred from "sequences of integer tokens" and chunked token indexing; `Capped` is inferred from explicit sequence/chunk limits ("We use n = 2048 and m = 64."); `Dynamic` attention is inferred from runtime nearest-neighbor retrieval ("For each chunk C, we retrieve its approximate k-nearest neighbours..."); `Constructed` state is inferred from building encoded neighbor state E ("the retrieved tokens Ret(C) are fed into an encoder Transformer, which computes the encoded neighbours set E.").

### Task: Open-domain question answering
- "We fine-tune our retrieval models on the Natural Questions (Kwiatkowski et al., 2019) dataset" (Section 4.3. Question answering)
- "We format the data as \"question: {question} \nanswer: {answer}\"" (Section 4.3. Question answering)
- "The model has access to the question via the previous tokens in the sequence as well as the top 20 DPR Wikipedia passages and their titles via the chunked cross-attention mechanism." (Section 4.3. Question answering)
- "Table 5 | **Question answering results.** Exact match accuracy on Natural Questions." (Section 4.4. Relating retrieval performance to dataset leakage.)
- Inference: `1D (t)` is inferred because both question/passages and answers are token text; `Capped` input/output dynamics are inferred from fixed retrieval count ("top 20 retrieved passages") and chunked sequence formatting ("first chunk of 64 tokens"); `Static` attention is inferred for QA because passages are externally supplied from DPR as a fixed set for the model run; `Constructed` state is inferred because the model integrates encoded retrieval context through chunked cross-attention.
