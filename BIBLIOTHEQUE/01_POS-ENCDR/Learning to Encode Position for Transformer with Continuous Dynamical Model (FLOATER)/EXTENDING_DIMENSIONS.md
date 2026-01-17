## 1. Basic Metadata
- Title: "Learning to Encode Position for Transformer with Continuous Dynamical Model" (Title block)
- Authors: "Xuanqing Liu" (Title block); "Hsiang-Fu Yu" (Title block); "Inderjit Dhillon" (Title block); "Cho-Jui Hsieh" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary
The paper's primary contribution is FLOATER, described as "a new position encoder for Transformer, which models the position information via a continuous dynamical model in a data-driven and parameter-efficient manner" (1 Introduction).

## 3. Tasks Evaluated
- Task name: Neural machine translation (WMT14 En-De, WMT14 En-Fr). Task type: Other (machine translation). Dataset(s) used: WMT14 En-De; WMT14 En-Fr. Domain: natural language text. Evidence: "For neural machine translation problems (WMT14 En-De and En-Fr)" (B.1 Settings of ODE solver); "Experimental results of various position encoders on the machine translation task." (Table 2)
- Task name: GLUE benchmark (language understanding). Task type: Other (language understanding benchmark). Dataset(s) used: GLUE benchmark (eight datasets). Domain: natural language text. Evidence: "**GLUE benchmark** consists of eight datasets and each have different hyperparameter settings." (B.3 Training language understanding tasks); "We evaluate our new position layers on a variety of neural machine translation and language understanding tasks, the experimental results show consistent improvements over the baselines." (Abstract)
- Task name: SQuAD benchmark (question answering). Task type: Other (question answering / reading comprehension). Dataset(s) used: SQuAD. Domain: natural language text. Evidence: "**SQuAD benchmark.** For this benchmark we wrote our own finetuning code because currently there is no official code available." (B.3 Training language understanding tasks); "We demonstrate that FLOATER consistent improvements over baseline models across a variety of NLP tasks ranging from machine translations, language understanding, and question answering." (1 Introduction)
- Task name: RACE benchmark (reading comprehension). Task type: Other (reading comprehension / question answering). Dataset(s) used: RACE. Domain: natural language text. Evidence: "**RACE benchmark.** This benchmark has the longest context and sequence length." (B.3 Training language understanding tasks); "We demonstrate that FLOATER consistent improvements over baseline models across a variety of NLP tasks ranging from machine translations, language understanding, and question answering." (1 Introduction)

## 4. Domain and Modality Scope
- Evaluation scope: Multiple domains within the same modality (NLP text), as the paper reports "a variety of NLP tasks ranging from machine translations, language understanding, and question answering." (1 Introduction)
- Modality evidence: "natural language processing (NLP) tasks such as language modeling [4], neural machine translation (NMT) [1], and language understanding [2]." (1 Introduction)
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Neural machine translation (WMT14 En-De) | No (trained per dataset) | Yes (warm-start on same dataset) | Not specified. | "Train the original Transformer model for 30 epochs." (B.2 Training NMT tasks); "With the warm-initialized FLOWER checkpoint, retrain on the same dataset for 10 epochs (En-De) or 1 epoch (En-Fr)." (B.2 Training NMT tasks) |
| Neural machine translation (WMT14 En-Fr) | No (trained per dataset) | Yes (warm-start on same dataset) | Not specified. | "Train the original Transformer model for 30 epochs." (B.2 Training NMT tasks); "With the warm-initialized FLOWER checkpoint, retrain on the same dataset for 10 epochs (En-De) or 1 epoch (En-Fr)." (B.2 Training NMT tasks) |
| GLUE benchmark | Yes (pretrained RoBERTa initialization) | Yes | Not specified. | "For GLUE/SQuAD/RACE benchmarks, our experiments are all conducted upon RoBERTa" (B.3 Training language understanding tasks); "we initialize our FLOWER model with pretrained RoBERTa" (B.3 Training language understanding tasks); "When finetuning on GLUE datasets, we can choose to freeze the encoding layers." (4.4 Remarks on Training and Testing Efficiency) |
| SQuAD benchmark | Yes (pretrained RoBERTa initialization) | Yes | Not specified. | "For GLUE/SQuAD/RACE benchmarks, our experiments are all conducted upon RoBERTa" (B.3 Training language understanding tasks); "**SQuAD benchmark.** For this benchmark we wrote our own finetuning code because currently there is no official code available." (B.3 Training language understanding tasks) |
| RACE benchmark | Yes (pretrained RoBERTa initialization) | Yes (RoBERTa fine-tuned; w_h frozen) | Not specified. | "For GLUE/SQuAD/RACE benchmarks, our experiments are all conducted upon RoBERTa" (B.3 Training language understanding tasks); "In this benchmark we freeze the weights  w_h  and only finetune the weights of RoBERTa." (B.3 Training language understanding tasks) |

## 6. Input and Representation Constraints
- Variable-length sequences: "sequence data of variable lengths." (1 Introduction)
- Sequence length/embedding dimension: "L is the length of the sequence and d is the dimension of the word embedding." (2.1 Importance of Position Encoding for Transformer)
- Maximum length constraint for embedding-based PE: "the position embedding restricts the maximum length of input sequences." (Abstract)
- Fixed maximum length value used in prior/learned embeddings: "This data-driven approach comes at the cost of the limitation of a fixed maximum length of input sequence  $L_{\rm max}$  and the computational/memory overhead of additional  $L_{\rm max} \times d$  parameters, where  $L_{\rm max}$  is usually set to 512 in many applications, and d is the dimension of the embeddings." (1 Introduction)
- Discrete positions and equidistant assumption: "positions are discrete values as  $\{0,1,2,\dots\}$ ." (C Cases suitable for non-equidistant discritization); "By choosing positions t equidistantly, we are implicitly assuming the position signal evolves steadily as we go through each token in a sentence." (C Cases suitable for non-equidistant discritization)
- Fixed patch size / fixed number of tokens / padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure
- Maximum sequence length: "This data-driven approach comes at the cost of the limitation of a fixed maximum length of input sequence  $L_{\rm max}$  and the computational/memory overhead of additional  $L_{\rm max} \times d$  parameters, where  $L_{\rm max}$  is usually set to 512 in many applications, and d is the dimension of the embeddings." (1 Introduction)
- Fixed or variable length: "sequence data of variable lengths." (1 Introduction)
- Attention type: "Transformer utilizes a non-recurrent but self-attentive neural architecture to model the dependency among elements at different positions in the sequence" (1 Introduction). Windowed/hierarchical/sparse attention is not stated.
- Computational cost mechanisms: "we can 1) cache the positional bias vectors for some iterations without re-computing, 2) update the weights of flow models less frequently than other parts of the Transformer, and 3) update the flow models with a larger learning rate to accelerate convergence." (4.4 Remarks on Training and Testing Efficiency); "there is no overhead during the inference stage if we store the pre-calculated positional bias vectors in the checkpoints." (4.4 Remarks on Training and Testing Efficiency)

## 8. Positional Encoding (Critical Section)
- Positional encoding mechanism used: Absolute (index-based) learned via a continuous dynamical system; "we propose to use a dynamical system to model these position representations" (3.1 Position Encoding with Dynamical Systems); "We model the evolution of encoded results along position index by such a dynamical system" (Abstract).
- Where it is applied: "the position representation is integrated into each block in the hierarchy (there are N blocks in total)" (Figure 1); the paper also describes input-only injection as "inject position information only at the input block" (2.2 Position Encoding in Transformer) and evaluates settings with "position encoder at all blocks or only at the input block" (4.1).
- Fixed/modified/ablated: The paper compares multiple encoders, including "FLOATER", "Pre-defined Sinusoidal Position Encoder", and "Fixed-length Position Embedding" (Table 2), and also explores an alternative where "we model the sequence  $\{p_i\}_{i\in\{1,2,\dots\}}$  with RNN models" (Is RNN a good alternative to model the dynamics?).

## 9. Positional Encoding as a Variable
- Core research variable: Yes, the work centers on a new positional encoding: "We propose FLOATER, a new position encoder for Transformer" (1 Introduction).
- Multiple positional encodings compared: Yes; "Experimental results of various position encoders on the machine translation task." (Table 2); "To see if RNN works equally well, we model the sequence  $\{p_i\}_{i\in\{1,2,\dots\}}$  with RNN models" (Is RNN a good alternative to model the dynamics?).
- Claim that PE choice is not critical or secondary: Not stated.

## 10. Evidence of Constraint Masking
- Model sizes: "Typically there are 6 blocks in sequence-to-sequence Transformer and 12 or 24 blocks in BERT." (3.2 Parameter Sharing among Blocks); "both *Transformer-base* and *Transformer-large* models" (4.1).
- Dataset sizes: Not specified.
- Attribution of gains: Improvements are attributed to the proposed positional encoding rather than scaling, e.g., "the experimental results show consistent improvements over the baselines." (Abstract).

## 11. Architectural Workarounds
- Parameter sharing across blocks to limit parameters: "we address this issue by sharing parameters across all the blocks" (3.2 Parameter Sharing among Blocks).
- Multi-layer position injection: "the position representation is integrated into each block in the hierarchy (there are N blocks in total)" (Figure 1); "position encoder at all blocks or only at the input block" (4.1).
- Training efficiency optimizations: "Initialize with pretrained models that do not contain flow-based dynamics" (4.4 Remarks on Training and Testing Efficiency); "we can 1) cache the positional bias vectors for some iterations without re-computing, 2) update the weights of flow models less frequently than other parts of the Transformer, and 3) update the flow models with a larger learning rate to accelerate convergence." (4.4 Remarks on Training and Testing Efficiency).
- RoBERTa integration and freezing: "we first download a pretrained RoBERTa model, plug in some flow-based encoding layers, and re-train the encoding layers on WikiText-103 dataset for one epoch. When finetuning on GLUE datasets, we can choose to freeze the encoding layers." (4.4 Remarks on Training and Testing Efficiency); "In this benchmark we freeze the weights  w_h  and only finetune the weights of RoBERTa." (B.3 Training language understanding tasks).

## 12. Explicit Limitations and Non-Claims
- Training overhead: "our flow-based method adds a non-negligible time and memory overhead" (4.4 Remarks on Training and Testing Efficiency).
- Data size limitation for dynamics training: "GLUE/SQuAD/RACE datasets are too small to train dynamics from scratch" (B.3 Training language understanding tasks).
- Future work / non-coverage: "In this paper, we are not going to explore the more general cases discussed above. Instead, we decided to leave them as interesting future work." (C Cases suitable for non-equidistant discritization)
- Non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)
> **Constraint Profile:**
> - Domain scope: Multiple NLP tasks within text modality ("a variety of NLP tasks ranging from machine translations, language understanding, and question answering." (1 Introduction)).
> - Task structure: Benchmarks are specific NLP datasets like "WMT14 En-De and En-Fr" and "GLUE/SQuAD/RACE benchmarks" (B.1; B.3).
> - Representation rigidity: Variable-length sequences with discrete positions ("sequence data of variable lengths." (1 Introduction); "positions are discrete values as  $\{0,1,2,\dots\}$ ." (C)).
> - Model sharing vs specialization: NMT is trained per dataset ("retrain on the same dataset for 10 epochs (En-De) or 1 epoch (En-Fr)." (B.2)), while GLUE/SQuAD/RACE are initialized from a shared pretrained backbone ("we initialize our FLOWER model with pretrained RoBERTa" (B.3)).
> - Role of positional encoding: Central variable compared across encoders ("Experimental results of various position encoders on the machine translation task." (Table 2)).

### 14. Final Classification
Classification: **Multi-task, single-domain**.
The paper evaluates multiple NLP tasks, stating it covers "a variety of NLP tasks ranging from machine translations, language understanding, and question answering." (1 Introduction). The benchmarks named are all text datasets such as "WMT14 En-De and En-Fr" and "GLUE/SQuAD/RACE benchmarks" (B.1; B.3), with no multi-modal or cross-domain transfer claims.
