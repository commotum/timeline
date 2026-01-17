## 1. Basic Metadata
- Title: "DEBERTA: DECODING-ENHANCED BERT WITH DIS-ENTANGLED ATTENTION" (Title)
- Authors: "Pengcheng He<sup>1</sup>, Xiaodong Liu<sup>2</sup>, Jianfeng Gao<sup>2</sup>, Weizhu Chen<sup>1</sup>" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary
DeBERTa proposes a Transformer-based pre-trained language model that improves BERT and RoBERTa by introducing disentangled attention and an enhanced mask decoder to better use content and position information for downstream NLP tasks.

## 3. Tasks Evaluated
- Task name: CoLA (GLUE)
  - Task type: Classification
  - Dataset(s) used: CoLA
  - Domain: Natural language text
  - Evidence:
    > "| CoLA       | Acceptability   | 8.5k      | 1k        | 1k        | 2      | Matthews corr         |  |  |" (A.1 Dataset, Table 6)
- Task name: SST (SST-2)
  - Task type: Classification
  - Dataset(s) used: SST
  - Domain: Natural language text
  - Evidence:
    > "| SST        | Sentiment       | 67k       | 872       | 1.8k      | 2      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Task name: MNLI
  - Task type: Classification; Reasoning / relational
  - Dataset(s) used: MNLI
  - Domain: Natural language text
  - Evidence:
    > "| MNLI       | NLI             | 393k      | 20k       | 20k       | 3      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Task name: QQP
  - Task type: Classification
  - Dataset(s) used: QQP
  - Domain: Natural language text
  - Evidence:
    > "| QQP        | Paraphrase      | 364k      | 40k       | 391k      | 2      | Accuracy/F1           |  |  |" (A.1 Dataset, Table 6)
- Task name: MRPC
  - Task type: Classification
  - Dataset(s) used: MRPC
  - Domain: Natural language text
  - Evidence:
    > "| MRPC       | Paraphrase      | 3.7k      | 408       | 1.7k      | 2      | Accuracy/F1           |  |  |" (A.1 Dataset, Table 6)
- Task name: QNLI
  - Task type: Classification
  - Dataset(s) used: QNLI
  - Domain: Natural language text
  - Evidence:
    > "| QNLI       | QA/NLI          | 108k      | 5.7k      | 5.7k      | 2      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Task name: STS-B
  - Task type: Other (similarity/regression)
  - Dataset(s) used: STS-B
  - Domain: Natural language text
  - Evidence:
    > "| STS-B      | Similarity      | 7k        | 1.5k      | 1.4k      | 1      | Pearson/Spearman corr |  |  |" (A.1 Dataset, Table 6)
- Task name: RTE
  - Task type: Classification; Reasoning / relational
  - Dataset(s) used: RTE
  - Domain: Natural language text
  - Evidence:
    > "| RTE        | NLI             | 2.5k      | 276       | 3k        | 2      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Task name: SQuAD v1.1
  - Task type: Other (machine reading comprehension / extractive QA)
  - Dataset(s) used: SQuAD v1.1
  - Domain: Natural language text
  - Evidence:
    > "| SQuAD v1.1 | MRC             | 87.6k     | 10.5k     | 9.5k      | -      | Exact Match (EM)/F1   |  |  |" (A.1 Dataset, Table 6)
- Task name: SQuAD v2.0
  - Task type: Other (machine reading comprehension / extractive QA)
  - Dataset(s) used: SQuAD v2.0
  - Domain: Natural language text
  - Evidence:
    > "| SQuAD v2.0 | MRC             | 130.3k    | 11.9k     | 8.9k      | -      | Exact Match (EM)/F1   |  |  |" (A.1 Dataset, Table 6)
- Task name: RACE
  - Task type: Other (machine reading comprehension / multiple-choice)
  - Dataset(s) used: RACE
  - Domain: Natural language text
  - Evidence:
    > "RACE is a large-scale machine reading comprehension dataset, collected from English examinations in China, which are designed for middle school and high school students (Lai et al., 2017)." (A.1 Dataset)
- Task name: ReCoRD
  - Task type: Other (machine reading comprehension / QA)
  - Dataset(s) used: ReCoRD
  - Domain: Natural language text
  - Evidence:
    > "| ReCoRD     | MRC             | 101k      | 10k       | 10k       | -      | Exact Match (EM)/F1   |  |  |" (A.1 Dataset, Table 6)
- Task name: SWAG
  - Task type: Classification; Reasoning / relational
  - Dataset(s) used: SWAG
  - Domain: Natural language text
  - Evidence:
    > "SWAG is a large-scale adversarial dataset for the task of grounded commonsense inference, which unifies natural language inference and physically grounded reasoning (Zellers et al., 2018). SWAG consists of 113k multiple choice questions about grounded situations." (A.1 Dataset)
- Task name: CoNLL 2003 NER
  - Task type: Other (token classification / NER)
  - Dataset(s) used: CoNLL 2003
  - Domain: Natural language text
  - Evidence:
    > "| CoNLL 2003 | NER             | 14,987    | 3,466     | 3,684     | 8      | F1                    |  |  |" (A.1 Dataset, Table 6)
    > "CoNLL 2003 is an English dataset consisting of text from a wide variety of sources. It has 4 types of named entity." (A.1 Dataset)
- Task name: BoolQ
  - Task type: Classification
  - Dataset(s) used: BoolQ
  - Domain: Natural language text
  - Evidence:
    > "| BoolQ      | QA              | 9,427     | 3,270     | 3,245     | 2      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Task name: COPA
  - Task type: Classification; Reasoning / relational
  - Dataset(s) used: COPA
  - Domain: Natural language text
  - Evidence:
    > "| COPA       | QA              | 400k      | 100       | 500       | 2      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Task name: CB (listed as "СВ" in Table 6)
  - Task type: Classification; Reasoning / relational
  - Dataset(s) used: CB
  - Domain: Natural language text
  - Evidence:
    > "| СВ         | NLI             | 250       | 57        | 250       | 3      | Accuracy/F1           |  |  |" (A.1 Dataset, Table 6)
- Task name: MultiRC
  - Task type: Other (multiple-choice QA)
  - Dataset(s) used: MultiRC
  - Domain: Natural language text
  - Evidence:
    > "| MultiRC    | Multiple choice | 5,100     | 953       | 1,800     | -      | Exact Match (EM)/F1   |  |  |" (A.1 Dataset, Table 6)
- Task name: WiC
  - Task type: Classification
  - Dataset(s) used: WiC
  - Domain: Natural language text
  - Evidence:
    > "| WiC        | WSD             | 2.5k      | 276       | 3k        | 2      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Task name: WSC
  - Task type: Classification; Reasoning / relational
  - Dataset(s) used: WSC
  - Domain: Natural language text
  - Evidence:
    > "| WSC        | Coreference     | 554k      | 104       | 146       | 2      | Accuracy              |  |  |" (A.1 Dataset, Table 6)
- Task name: Auto-regressive language modeling (ARLM) on Wikitext-103
  - Task type: Generation
  - Dataset(s) used: Wikitext-103
  - Domain: Natural language text
  - Evidence:
    > "We evaluate DeBERTa on the task of auto-regressive language model (ARLM) using Wikitext-103 (Merity et al., 2016)." (A.4 Main Results on Generation Tasks)

## 4. Domain and Modality Scope
- Evaluation scope: Multiple datasets within the same modality (text), spanning NLU and NLG tasks. Evidence: "the performance of both natural language understand (NLU) and natural language generation (NLG) downstream tasks." (Abstract)
- Single domain vs multiple domains: Multiple NLP benchmarks/tasks rather than a single dataset. Evidence: "We summarize the results on eight NLU tasks of GLUE (Wang et al., 2019b) in Table 1" (5.1.1 Performance on Large Models).
- Multiple modalities: Not reported; the paper discusses "natural language processing (NLP) tasks." (Abstract)
- Domain generalization or cross-domain transfer: Not claimed. The paper only states "a new virtual adversarial training method is used for fine-tuning to improve models' generalization." (Abstract)

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| CoLA | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| SST (SST-2) | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| MNLI | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| QQP | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| MRPC | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| QNLI | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| STS-B | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| RTE | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| SQuAD v1.1 | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| SQuAD v2.0 | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| RACE | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| ReCoRD | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| SWAG | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| CoNLL 2003 NER | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| BoolQ | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| COPA | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| CB ("СВ") | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| MultiRC | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| WiC | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| WSC | No (per-task fine-tuning) | Yes | Not specified. | "For fine-tuning, we train each task with a hyper-parameter search procedure, each run takes about 1-2 hours on a DGX-2 node." (A.3 Implementation Details) |
| Auto-regressive language modeling (ARLM) on Wikitext-103 | Joint MLM+ARLM pre-training (DeBERTa-MT) | Not specified. | Not specified. | "It is jointly pre-trained using the MLM and ARLM tasks as in UniLM (Dong et al., 2019)." (A.4 Main Results on Generation Tasks) |

## 6. Input and Representation Constraints
- Sequence length is variable (N): "N the length of the input sequence." (3.1 Disentangled Attention)
- Relative distance is truncated and bounded by k: "Denote k as the maximum relative distance,  $\delta(i, j) \in [0, 2k)$  as the relative distance from token i to token j" (3.1 Disentangled Attention).
- Explicit maximum relative distance used in experiments: "In our experiments, we set the maximum relative distance k to 512 for pre-training." (3.1.1 Efficient Implementation)
- Masking procedure for MLM inputs: "we corrupt it into  $\tilde{X}$  by masking 15% of its tokens at random" (2.2 Masked Language Model) and "The authors of BERT propose to keep 10% of the masked tokens unchanged, another 10% replaced with randomly picked tokens and the rest replaced with the [MASK] token." (2.2 Masked Language Model)
- Tokenization/vocabulary constraints: "we use the BPE vocabulary of Radford et al. (2019); Liu et al. (2019c)." (5.1.1 Performance on Large Models) and "a new vocabulary of size 128K constructed using the dataset." (5.3 Scale Up to 1.5 Billion Parameters)
- Fine-tuning perturbations operate on embeddings: "For NLP tasks, the perturbation is applied to the word embedding instead of the original word sequence." (4 Scale Invariant Fine-Tuning)
- Autoregressive generation mask: "we use a triangular matrix for self-attention and set the upper triangular part of the self-attention mask to  $-\infty$" (A.4 Main Results on Generation Tasks)

## 7. Context Window and Attention Structure
- Maximum sequence length (theoretical, with k and L specified): "Taking DeBERTa $_{large}$  as an example, where k=512, L=24, in theory, the maximum sequence length that can be handled is 24,528." (A.5 Handling Long Sequence Input)
- Sequence length is variable: "N the length of the input sequence." (3.1 Disentangled Attention)
- Attention type: multi-head self-attention with relative position bias, and truncated relative distance. Evidence: "Each block contains a multi-head self-attention layer" (2.1 Transformer) and "With relative position bias, we choose to truncate the maximum relative distance to k as in equation 3. Thus in each layer, each token can attend directly to at most 2(k-1) tokens and itself." (A.5 Handling Long Sequence Input)
- Computational cost management for relative positions: "we do not need to allocate memory to store a relative position embedding for each query and thus reduce the space complexity to O(kd)" (3.1.1 Efficient Implementation)
- Autoregressive attention mask for NLG: "we use a triangular matrix for self-attention and set the upper triangular part of the self-attention mask to  $-\infty$" (A.4 Main Results on Generation Tasks)

## 8. Positional Encoding (Critical Section)
- Mechanism and relative positions: "each word in DeBERTa is represented using two vectors that encode its content and position, respectively, and the attention weights among words are computed using disentangled matrices based on their contents and relative positions, respectively." (Introduction)
- Relative position embeddings are shared across layers: "P in R^{2k\times d} represents the relative position embedding vectors shared across all layers (i.e., staying fixed during forward propagation)" (3.1 Disentangled Attention)
- Absolute positions added in decoding layer (EMD): "In DeBERTa, we incorporate them right after all the Transformer layers but before the *softmax* layer for masked token prediction" (3.2 Enhanced Mask Decoder Accounts for Absolute Word Positions)
- Comparison against input-layer absolute positions: "The BERT model incorporates absolute positions in the input layer." (3.2 Enhanced Mask Decoder Accounts for Absolute Word Positions) and "DeBERTa-AP is a variant of DeBERTa where absolute position embeddings are incorporated in the input layer as RoBERTa." (A.4 Main Results on Generation Tasks)
- PE variation/ablation: "In the empirical study, we compare these two methods of incorporating absolute positions and observe that EMD works much better." (3.2 Enhanced Mask Decoder Accounts for Absolute Word Positions)

## 9. Positional Encoding as a Variable
- Core research variable: Yes. The paper frames its main contributions as positional mechanisms: "two novel techniques: a disentangled attention mechanism, and an enhanced mask decoder." (Introduction)
- Multiple positional encodings compared: Yes. "we compare these two methods of incorporating absolute positions" (3.2 Enhanced Mask Decoder Accounts for Absolute Word Positions) and "DeBERTa-AP is a variant of DeBERTa where absolute position embeddings are incorporated in the input layer as RoBERTa." (A.4 Main Results on Generation Tasks)
- Claim that PE choice is not critical: Not stated.

## 10. Evidence of Constraint Masking
- Model sizes: "we scale up DeBERTa by training a larger version that consists of 48 Transform layers with 1.5 billion parameters." (Abstract) and "The model consists of 48 layers with a hidden size of 1,536 and 24 attention heads" (5.3 Scale Up to 1.5 Billion Parameters)
- Dataset sizes: "The total data size after data deduplication (Shoeybi et al., 2019) is about 78G." (5.1.1 Performance on Large Models) and "DeBERTa<sub>1.5B</sub> is trained on a pre-training dataset amounting to 160G" (5.3 Scale Up to 1.5 Billion Parameters)
- Performance attributed to scaling model size: "The significant performance boost due to scaling DeBERTa to a larger model makes the single DeBERTa<sub>1.5B</sub> surpass the human performance on SuperGLUE" (5.3 Scale Up to 1.5 Billion Parameters)
- Performance attributed to architectural techniques: "two novel techniques. The first is the disentangled attention mechanism... Second, an enhanced mask decoder is used" (Abstract)
- Training tricks/generalization: "a new virtual adversarial training method is used for fine-tuning to improve models' generalization." (Abstract)
- Data scaling context: "a DeBERTa model trained on half of the training data performs consistently better" (Abstract)

## 11. Architectural Workarounds
- Relative distance truncation for long-sequence handling: "we choose to truncate the maximum relative distance to k" and "each token can attend directly to at most 2(k-1) tokens and itself." (A.5 Handling Long Sequence Input)
- Memory/computation optimization for relative positions: "we do not need to allocate memory to store a relative position embedding for each query and thus reduce the space complexity to O(kd)" (3.1.1 Efficient Implementation)
- Parameter reduction via projection sharing: "we share the projection matrices of relative position embedding  $W_{k,r}$ ,  $W_{q,r}$  with  $W_{k,c}$ ,  $W_{q,c}$ , respectively, in all attention layers to reduce the number of model parameters." (5.3 Scale Up to 1.5 Billion Parameters)
- Convolutional augmentation for sub-word n-gram knowledge: "a convolution layer is added aside the first Transformer layer to induce n-gram knowledge of sub-word encodings" (5.3 Scale Up to 1.5 Billion Parameters)
- Decoding-layer absolute positions (EMD): "we incorporate them right after all the Transformer layers but before the *softmax* layer for masked token prediction" (3.2 Enhanced Mask Decoder Accounts for Absolute Word Positions)
- Autoregressive mask for generation: "we use a triangular matrix for self-attention and set the upper triangular part of the self-attention mask to  $-\infty$" (A.4 Main Results on Generation Tasks)
- Shared EMD weights to reduce parameters: "In our experiment we share the same weight for n=2 layers to reduce the number of parameters" (A.8 Additional Details of Enhanced Mask Decoder)

## 12. Explicit Limitations and Non-Claims
- Limited SiFT study scope: "Note that we **only** apply SiFT to DeBERTa<sub>1.5B</sub> on SuperGLUE tasks in our experiments and we will provide a more comprehensive study of SiFT in our future work." (4 Scale Invariant Fine-Tuning)
- Future work on enhanced decoder inputs: "EMD also enables us to introduce other useful information, in addition to positions, for pre-training. We leave it to future work." (3.2 Enhanced Mask Decoder Accounts for Absolute Word Positions)
- Long-sequence limitation/future work: "One of our future research directions is to extend DeBERTa to deal with extremely long sequences." (A.5 Handling Long Sequence Input)
- Not a claim of human-level NLU: "Despite its promising results on SuperGLUE, the model is by no means reaching the human-level intelligence of NLU." (6 Conclusions)
- Future work on compositional generalization: "Moving forward, it is worth exploring how to make DeBERTa incorporate compositional structures in a more explicit manner" (6 Conclusions)

### 13. Constraint Profile (Synthesis)
> **Constraint Profile:**
> - Domain scope: Multiple NLP benchmarks and tasks in a single modality (text), spanning NLU and NLG.
> - Task structure: Many supervised NLU benchmarks plus an autoregressive language modeling evaluation; no open-ended or cross-modal tasks described.
> - Representation rigidity: Token-based text with BPE vocabulary and truncated relative distance (k=512), with explicit sequence-length handling for long inputs.
> - Model sharing vs specialization: Per-task fine-tuning for NLU tasks; a separate DeBERTa-MT variant is jointly pre-trained for ARLM.
> - Role of positional encoding: Central design variable (disentangled relative positions plus absolute positions in the decoding layer, compared against input-layer absolute positions).

### 14. Final Classification
**Multi-task, single-domain.** The paper evaluates across many NLP tasks within text only, e.g., "We summarize the results on eight NLU tasks of GLUE" (5.1.1 Performance on Large Models) and also reports NLG results ("In addition to NLU tasks, DeBERTa can also be extended to handle NLG tasks." A.4 Main Results on Generation Tasks). The domain is consistently natural language ("natural language processing (NLP) tasks." Abstract), with no multi-modal evaluation or cross-domain transfer claims.
