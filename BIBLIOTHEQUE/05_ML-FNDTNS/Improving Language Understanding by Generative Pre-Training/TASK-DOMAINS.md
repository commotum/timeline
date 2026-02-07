# Improving Language Understanding by Generative Pre-Training (Not specified in the paper)
Source: Improving Language Understanding by Generative Pre-Training.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling (next-token prediction) | tokens (context tokens) | 1D (t) | Fixed | Static (inferred) | Constructed (inferred) | tokens (next-token distribution) | 1D (t) | Fixed |
| Natural language inference (textual entailment classification) | premise and hypothesis tokens | 1D (t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | entailment/contradiction/neutral label | 0D | Fixed |
| Question answering (multiple-choice reading comprehension) | document context + question + candidate answer tokens | 1D (t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | answer choice label | 0D | Fixed |
| Commonsense reasoning / story completion (ending selection) | story context + candidate ending tokens | 1D (t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | correct ending choice label | 0D | Fixed |
| Semantic similarity / paraphrase detection (binary classification) | sentence pair tokens | 1D (t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | equivalence label | 0D | Fixed |
| Sentiment analysis (text classification) | sentence tokens | 1D (t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | sentiment label (positive/negative) | 0D | Fixed |
| Linguistic acceptability (grammaticality classification) | sentence tokens | 1D (t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | grammaticality label | 0D | Fixed |
| Winograd schemas / pronoun resolution (coreference selection) | sentence with pronoun alternatives tokens | 1D (t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | resolution choice label | 0D | Fixed |

## Summary
The paper covers a suite of text-only NLP tasks spanning generative language modeling and multiple discriminative classification problems. Inputs are token sequences (single, paired, or structured concatenations), and outputs are mostly single labels, with language modeling producing token predictions. The model processes fixed-length sequences for pre-training and (by the same context window) capped-length inputs for fine-tuned tasks, with static attention over the provided context and a transformer-based constructed internal state (inferred from the architecture description).

## Evidence
### Task: Language modeling (next-token prediction)
- "Given an unsupervised corpus of tokens  $\mathcal{U} = \{u_1, \dots, u_n\}$ , we use a standard language modeling objective" (Section 3.1 Unsupervised pre-training)
- "produce an output distribution over target tokens" (Section 3.1 Unsupervised pre-training)
- "contiguous sequences of 512 tokens." (Section 4.1 Setup, Model specifications)
- Inference: Attention Dynamic and State Dynamic marked Static/Constructed (inferred) because the model applies "multi-headed self-attention operation over the input context tokens" and provides "a more structured memory for handling long-term dependencies in text." (Section 3.1 Unsupervised pre-training; Introduction)

### Task: Natural language inference (textual entailment classification)
- "involves reading a pair of sentences and judging the relationship between them from one of entailment, contradiction or neutral." (Section 4.2 Supervised fine-tuning)
- "we concatenate the premise p and hypothesis h token sequences, with a delimiter token ($) in between." (Section 3.3 Task-specific input transformations)
- Inference: In Dynamics marked Capped (inferred) because inputs are processed as a "single contiguous sequence of tokens" with a finite "context window"; Attention/State marked Static/Constructed (inferred) from "multi-headed self-attention operation over the input context tokens" and "structured memory." (Introduction; Section 3.1)

### Task: Question answering (multiple-choice reading comprehension)
- "we are given a context document z, a question q, and a set of possible answers  $\{a_k\}$ ." (Section 3.3 Task-specific input transformations)
- "English passages with associated questions" (Section 4.2 Supervised fine-tuning)
- "output distribution over possible answers." (Section 3.3 Task-specific input transformations)
- Inference: In Dynamics marked Capped (inferred) because inputs are processed as a "single contiguous sequence of tokens" with a finite "context window"; Attention/State marked Static/Constructed (inferred) from "multi-headed self-attention operation over the input context tokens" and "structured memory." (Introduction; Section 3.1)

### Task: Commonsense reasoning / story completion (ending selection)
- "Story Cloze Test [40], which involves selecting the correct ending to multi-sentence stories from two options." (Section 4.2 Supervised fine-tuning)
- "we are given a context document z, a question q, and a set of possible answers  $\{a_k\}$ ." (Section 3.3 Task-specific input transformations)
- Inference: In Dynamics marked Capped (inferred) because inputs are processed as a "single contiguous sequence of tokens" with a finite "context window"; Attention/State marked Static/Constructed (inferred) from "multi-headed self-attention operation over the input context tokens" and "structured memory." (Introduction; Section 3.1)

### Task: Semantic similarity / paraphrase detection (binary classification)
- "Semantic similarity (or paraphrase detection) tasks involve predicting whether two sentences are semantically equivalent or not." (Section 4.2 Supervised fine-tuning)
- "there is no inherent ordering of the two sentences being compared." (Section 3.3 Task-specific input transformations)
- Inference: In Dynamics marked Capped (inferred) because inputs are processed as a "single contiguous sequence of tokens" with a finite "context window"; Attention/State marked Static/Constructed (inferred) from "multi-headed self-attention operation over the input context tokens" and "structured memory." (Introduction; Section 3.1)

### Task: Sentiment analysis (text classification)
- "For SST-2 (sentiment analysis), we append the token *very* to each example" (Section 5 Analysis, Zero-shot Behaviors)
- "only the words *positive* and *negative*" (Section 5 Analysis, Zero-shot Behaviors)
- "The Stanford Sentiment Treebank (SST-2) [54], on the other hand, is a standard binary classification task." (Section 4.2 Supervised fine-tuning)
- Inference: In Dynamics marked Capped (inferred) because inputs are processed as a "single contiguous sequence of tokens" with a finite "context window"; Attention/State marked Static/Constructed (inferred) from "multi-headed self-attention operation over the input context tokens" and "structured memory." (Introduction; Section 3.1)

### Task: Linguistic acceptability (grammaticality classification)
- "The Corpus of Linguistic Acceptability (CoLA) [65] contains expert judgements on whether a sentence is grammatical or not" (Section 4.2 Supervised fine-tuning)
- "For CoLA (linguistic acceptability), examples are scored as the average token log-probability" (Section 5 Analysis, Zero-shot Behaviors)
- Inference: In Dynamics marked Capped (inferred) because inputs are processed as a "single contiguous sequence of tokens" with a finite "context window"; Attention/State marked Static/Constructed (inferred) from "multi-headed self-attention operation over the input context tokens" and "structured memory." (Introduction; Section 3.1)

### Task: Winograd schemas / pronoun resolution (coreference selection)
- "For DPRD [46] (winograd schemas), we replace the definite pronoun with the two possible referrents and predict the resolution" (Section 5 Analysis, Zero-shot Behaviors)
- "use the underlying generative model to perform tasks without supervised finetuning." (Section 5 Analysis, Zero-shot Behaviors)
- Inference: In Dynamics marked Capped (inferred) because inputs are processed as a "single contiguous sequence of tokens" with a finite "context window"; Attention/State marked Static/Constructed (inferred) from "multi-headed self-attention operation over the input context tokens" and "structured memory." (Introduction; Section 3.1)
