# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Not specified in the paper.)
Source: BERT- Pre-training of Deep Bidirectional Transformers for Language Understanding.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Masked language modeling (MLM) | token sequence with masked WordPiece tokens | 1D (t) (inferred) | Capped | Static (inferred) | Constructed (inferred) | masked token identities | 1D (t) (inferred) | Capped (inferred) |
| Next sentence prediction (NSP) | sentence pair (A and B) token sequence | 1D (t) (inferred) | Capped | Static (inferred) | Constructed (inferred) | IsNext/NotNext label | 0D (inferred) | Fixed (inferred) |
| Entailment classification (MNLI) | sentence pair | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | entailment/contradiction/neutral label | 0D (inferred) | Fixed (inferred) |
| Question pair equivalence classification (QQP) | question pair | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | equivalence label | 0D (inferred) | Fixed (inferred) |
| Answer sentence classification (QNLI) | question and sentence pair | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | contains-answer label | 0D (inferred) | Fixed (inferred) |
| Sentiment classification (SST-2) | single sentence | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | sentiment label | 0D (inferred) | Fixed (inferred) |
| Acceptability classification (CoLA) | single sentence | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | acceptable/unacceptable label | 0D (inferred) | Fixed (inferred) |
| Semantic textual similarity scoring (STS-B) | sentence pair | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | similarity score (1-5) | 0D (inferred) | Fixed (inferred) |
| Paraphrase classification (MRPC) | sentence pair | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | equivalence label | 0D (inferred) | Fixed (inferred) |
| Textual entailment classification (RTE) | sentence pair | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | entailment label | 0D (inferred) | Fixed (inferred) |
| Extractive question answering (SQuAD v1.1) | question and passage | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | answer text span | 1D (t) (inferred) | Capped (inferred) |
| Extractive QA with unanswerable option (SQuAD v2.0) | question and paragraph | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | answer span or no-answer | 1D (t) (inferred) | Capped (inferred) |
| Sentence continuation selection (SWAG) | sentence plus candidate continuations (4 sequences) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | choice label (most plausible continuation) | 0D (inferred) | Fixed (inferred) |
| Named entity recognition tagging (CoNLL-2003) | token sequence (document context) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | token-level NER labels | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper covers two pre-training objectives (masked language modeling and next sentence prediction) and a broad set of downstream NLP tasks spanning GLUE sentence/pair classification and similarity, extractive QA (SQuAD v1.1/v2.0), SWAG continuation selection, and CoNLL-2003 NER. Inputs are described as token sequences (single sentences, sentence pairs, question-passage pairs), with pre-training sequences explicitly capped at 512 tokens and downstream sequence lengths inferred to be capped by the same interface. Outputs are either fixed labels/scores (0D) or token-level spans/labels (1D (t)), while Attention and State dynamics are inferred from the self-attention architecture and contextual token representations.

## Evidence
### Task: Masked language modeling (MLM)
- "we simply mask some percentage of the input tokens at random, and then predict those masked tokens." (Section 3.1 Pre-training BERT)
- "They are sampled such that the combined length is  $\leq 512$  tokens." (Section A.2 Pre-training Procedure)
- Inference: Labeled In/Out Dimension, Attention Dynamic, State Dynamic, and Out Dynamics as inferred based on the description of BERT operating on an input token sequence with self-attention and contextual token representations C and T_i (Input/Output Representations; Section 3.2).

### Task: Next sentence prediction (NSP)
- "we pre-train for a binarized next sentence prediction task that can be trivially generated from any monolingual corpus." (Section 3.1 Pre-training BERT)
- "50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext)." (Section 3.1 Pre-training BERT)
- "They are sampled such that the combined length is  $\leq 512$  tokens." (Section A.2 Pre-training Procedure)
- Inference: Labeled In/Out Dimension, Attention Dynamic, State Dynamic, and Out Dynamics as inferred based on the description of BERT operating on an input token sequence with self-attention and contextual token representations C and T_i (Input/Output Representations; Section 3.2).

### Task: Entailment classification (MNLI)
- "MNLI Multi-Genre Natural Language Inference is a large-scale, crowdsourced entailment classification task (Williams et al., 2018). Given a pair of sentences, the goal is to predict whether the second sentence is an entailment, contradiction, or neutral with respect to the first one." (Section B.1 Detailed Descriptions for the GLUE Benchmark Experiments)
- Inference: Labeled In/Out Dimension and Dynamics, Attention Dynamic, and State Dynamic as inferred based on the description of BERT operating on an input token sequence with self-attention and contextual token representations C and T_i (Input/Output Representations; Section 3.2).

### Task: Question pair equivalence classification (QQP)
- "QQP Quora Question Pairs is a binary classification task where the goal is to determine if two questions asked on Quora are semantically equivalent (Chen et al., 2018)." (Section B.1 Detailed Descriptions for the GLUE Benchmark Experiments)
- Inference: Labeled In/Out Dimension and Dynamics, Attention Dynamic, and State Dynamic as inferred based on the description of BERT operating on an input token sequence with self-attention and contextual token representations C and T_i (Input/Output Representations; Section 3.2).

### Task: Answer sentence classification (QNLI)
- "QNLI Question Natural Language Inference is a version of the Stanford Question Answering Dataset (Rajpurkar et al., 2016) which has been converted to a binary classification task (Wang et al., 2018a)." (Section B.1 Detailed Descriptions for the GLUE Benchmark Experiments)
- "The positive examples are (question, sentence) pairs which do contain the correct answer, and the negative examples are (question, sentence) from the same paragraph which do not contain the answer." (Section B.1 Detailed Descriptions for the GLUE Benchmark Experiments)
- Inference: Labeled In/Out Dimension and Dynamics, Attention Dynamic, and State Dynamic as inferred based on the description of BERT operating on an input token sequence with self-attention and contextual token representations C and T_i (Input/Output Representations; Section 3.2).

### Task: Sentiment classification (SST-2)
- "SST-2 The Stanford Sentiment Treebank is a binary single-sentence classification task consisting of sentences extracted from movie reviews with human annotations of their sentiment (Socher et al., 2013)." (Section B.1 Detailed Descriptions for the GLUE Benchmark Experiments)
- Inference: Labeled In/Out Dimension and Dynamics, Attention Dynamic, and State Dynamic as inferred based on the description of BERT operating on an input token sequence with self-attention and contextual token representations C and T_i (Input/Output Representations; Section 3.2).

### Task: Acceptability classification (CoLA)
- "CoLA The Corpus of Linguistic Acceptability is a binary single-sentence classification task, where the goal is to predict whether an English sentence is linguistically \"acceptable\" or not (Warstadt et al., 2018)." (Section B.1 Detailed Descriptions for the GLUE Benchmark Experiments)
- Inference: Labeled In/Out Dimension and Dynamics, Attention Dynamic, and State Dynamic as inferred based on the description of BERT operating on an input token sequence with self-attention and contextual token representations C and T_i (Input/Output Representations; Section 3.2).

### Task: Semantic textual similarity scoring (STS-B)
- "STS-B The Semantic Textual Similarity Benchmark is a collection of sentence pairs drawn from news headlines and other sources (Cer et al., 2017). They were annotated with a score from 1 to 5 denoting how similar the two sentences are in terms of semantic meaning." (Section B.1 Detailed Descriptions for the GLUE Benchmark Experiments)
- Inference: Labeled In/Out Dimension and Dynamics, Attention Dynamic, and State Dynamic as inferred based on the description of BERT operating on an input token sequence with self-attention and contextual token representations C and T_i (Input/Output Representations; Section 3.2).

### Task: Paraphrase classification (MRPC)
- "MRPC Microsoft Research Paraphrase Corpus consists of sentence pairs automatically extracted from online news sources, with human annotations" (Section B.1 Detailed Descriptions for the GLUE Benchmark Experiments)
- "for whether the sentences in the pair are semantically equivalent (Dolan and Brockett, 2005)." (Section B.1 Detailed Descriptions for the GLUE Benchmark Experiments)
- Inference: Labeled In/Out Dimension and Dynamics, Attention Dynamic, and State Dynamic as inferred based on the description of BERT operating on an input token sequence with self-attention and contextual token representations C and T_i (Input/Output Representations; Section 3.2).

### Task: Textual entailment classification (RTE)
- "RTE Recognizing Textual Entailment is a binary entailment task similar to MNLI, but with much less training data (Bentivogli et al., 2009)." (Section B.1 Detailed Descriptions for the GLUE Benchmark Experiments)
- Inference: Labeled In/Out Dimension and Dynamics, Attention Dynamic, and State Dynamic as inferred based on the description of BERT operating on an input token sequence with self-attention and contextual token representations C and T_i (Input/Output Representations; Section 3.2).

### Task: Extractive question answering (SQuAD v1.1)
- "Given a question and a passage from Wikipedia containing the answer, the task is to predict the answer text span in the passage." (Section 4.2 SQuAD v1.1)
- Inference: Labeled In/Out Dimension and Dynamics, Attention Dynamic, and State Dynamic as inferred based on the description of BERT operating on an input token sequence with self-attention and contextual token representations C and T_i (Input/Output Representations; Section 3.2).

### Task: Extractive QA with unanswerable option (SQuAD v2.0)
- "The SQuAD 2.0 task extends the SQuAD 1.1 problem definition by allowing for the possibility that no short answer exists in the provided paragraph, making the problem more realistic." (Section 4.3 SQuAD v2.0)
- Inference: Labeled In/Out Dimension and Dynamics, Attention Dynamic, and State Dynamic as inferred based on the description of BERT operating on an input token sequence with self-attention and contextual token representations C and T_i (Input/Output Representations; Section 3.2).

### Task: Sentence continuation selection (SWAG)
- "Given a sentence, the task is to choose the most plausible continuation among four choices." (Section 4.4 SWAG)
- Inference: Labeled In/Out Dimension and Dynamics, Attention Dynamic, and State Dynamic as inferred based on the description of BERT operating on an input token sequence with self-attention and contextual token representations C and T_i (Input/Output Representations; Section 3.2).

### Task: Named entity recognition tagging (CoNLL-2003)
- "In this section, we compare the two approaches by applying BERT to the CoNLL-2003 Named Entity Recognition (NER) task (Tjong Kim Sang and De Meulder, 2003)." (Section 5.3 Feature-based Approach with BERT)
- "Following standard practice, we formulate this as a tagging task but do not use a CRF" (Section 5.3 Feature-based Approach with BERT)
- "We use the representation of the first sub-token as the input to the token-level classifier over the NER label set." (Section 5.3 Feature-based Approach with BERT)
- Inference: Labeled In/Out Dimension and Dynamics, Attention Dynamic, and State Dynamic as inferred based on the description of BERT operating on an input token sequence with self-attention and contextual token representations C and T_i (Input/Output Representations; Section 3.2).
