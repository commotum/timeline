## 1. Basic Metadata

Title: SELECTIVE ROTARY POSITION EMBEDDING. Quote: "SELECTIVE ROTARY POSITION EMBEDDING" (Document header).

Authors: Sajad Movahedi; Timur Carstensen; Arshia Afzal; Frank Hutter; Antonio Orvieto; Volkan Cevher. Quote: "Sajad Movahedi*<sup>1,4</sup>, Timur Carstensen*<sup>1,3</sup>, Arshia Afzal*<sup>2</sup>, Frank Hutter<sup>1,3,5</sup>, Antonio Orvieto<sup>†1,4</sup>, Volkan Cevher<sup>†2</sup>" (Document header).

Year: Year not specified.

Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

The paper proposes "Selective RoPE, an input-dependent rotary embedding mechanism, that generalizes RoPE" to improve recall by enabling input-dependent rotations for linear and softmax transformers in language modeling and sequence tasks (Abstract).

## 3. Tasks Evaluated

| Task name | Task type | Dataset(s) used | Domain | Evidence (quote + section) |
| --- | --- | --- | --- | --- |
| Multi-Query Associative Recall (MQAR) | Other (specify: associative recall / retrieval) | Multi-Query Associative Recall | Synthetic sequences | "We evaluate GLA + *Selective RoPE* on Multi-Query Associative Recall" (Section 4.2 Synthetic Language Tasks). |
| MAD - Compress | Other (specify: recall/sequence memory) | MAD benchmark suite | Synthetic sequences | "We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2 Synthetic Language Tasks). "| Model               | Compress    | Fuzzy<br>Recall | In-Context<br>Recall | Memorize | Noisy<br>Recall | Selective<br>Copy | Average |" (Table 1, Section 4.2). |
| MAD - Fuzzy Recall | Other (specify: recall/sequence memory) | MAD benchmark suite | Synthetic sequences | "We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2 Synthetic Language Tasks). "| Model               | Compress    | Fuzzy<br>Recall | In-Context<br>Recall | Memorize | Noisy<br>Recall | Selective<br>Copy | Average |" (Table 1, Section 4.2). |
| MAD - In-Context Recall | Other (specify: recall/sequence memory) | MAD benchmark suite | Synthetic sequences | "We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2 Synthetic Language Tasks). "| Model               | Compress    | Fuzzy<br>Recall | In-Context<br>Recall | Memorize | Noisy<br>Recall | Selective<br>Copy | Average |" (Table 1, Section 4.2). |
| MAD - Memorize | Other (specify: recall/sequence memory) | MAD benchmark suite | Synthetic sequences | "We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2 Synthetic Language Tasks). "| Model               | Compress    | Fuzzy<br>Recall | In-Context<br>Recall | Memorize | Noisy<br>Recall | Selective<br>Copy | Average |" (Table 1, Section 4.2). |
| MAD - Noisy Recall | Other (specify: recall/sequence memory) | MAD benchmark suite | Synthetic sequences | "We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2 Synthetic Language Tasks). "| Model               | Compress    | Fuzzy<br>Recall | In-Context<br>Recall | Memorize | Noisy<br>Recall | Selective<br>Copy | Average |" (Table 1, Section 4.2). |
| MAD - Selective Copy | Other (specify: recall/sequence memory) | MAD benchmark suite | Synthetic sequences | "We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2 Synthetic Language Tasks). "| Model               | Compress    | Fuzzy<br>Recall | In-Context<br>Recall | Memorize | Noisy<br>Recall | Selective<br>Copy | Average |" (Table 1, Section 4.2). |
| String copying | Other (specify: sequence copying) | String copying task | Synthetic sequences | "We also evaluate string copying following Jelassi et al. (2024). This task differs from *Selective Copy* in MAD in that the entire input sequence has to be copied token-by-token after the model is presented with a <copy> token." (Section 4.2 Synthetic Language Tasks). |
| State tracking on permutation composition ($S_2$) | Reasoning / relational | Permutation composition on $S_2$ | Synthetic sequences | "State Tracking. A common way to evaluate the expressivity of a model is *state tracking* on permutation composition (Liu et al., 2023)." (Section 4.2 Synthetic Language Tasks). "Figure 8: State tracking performance of GLA, Transformer, and DeltaNet with different positional embeddings on  $S_2$  and  $A_3$ ." (Figure 8 caption). |
| State tracking on permutation composition ($A_3$) | Reasoning / relational | Permutation composition on $A_3$ | Synthetic sequences | "State Tracking. A common way to evaluate the expressivity of a model is *state tracking* on permutation composition (Liu et al., 2023)." (Section 4.2 Synthetic Language Tasks). "Figure 8: State tracking performance of GLA, Transformer, and DeltaNet with different positional embeddings on  $S_2$  and  $A_3$ ." (Figure 8 caption). |
| Language modeling (pretraining) | Generation | FineWeb | Natural language text | "For our language modeling experiments we train 370M parameter versions of GLA (Yang et al., 2024a), Gated DeltaNet (Yang et al., 2025a), and the Forgetting Transformer (FoX) (Lin et al., 2025)" (Section 4.3 Language Modeling). "All models are trained on 35B tokens" (Section 4.3 Language Modeling). "of FineWeb (Penedo et al., 2024) at a context length of 4096 and use the Mistral 7B tokenizer (Jiang et al., 2023) with a vocabulary size of 32 000." (Section 4.3 Language Modeling). |
| Lambada (LMB.) | Generation; Classification | Lambada / LMB. | Natural language text | "For GLA, Selective RoPE reduces Lambada perplexity relative to RoPE and maintains comparable downstream accuracy to NoPE." (Section 4.3 Language Modeling). "| Model               | LMB.<br>ppl↓          | LMB.<br>acc ↑ | PIQA<br>acc ↑ | Hella.<br>acc_n ↑ | Wino.<br>acc ↑ | ARC-e<br>acc ↑ | ARC-c<br>acc_n ↑ | Avg.        |" (Table 2, Section 4.3). |
| PIQA | Classification | PIQA | Natural language text | "| Model               | LMB.<br>ppl↓          | LMB.<br>acc ↑ | PIQA<br>acc ↑ | Hella.<br>acc_n ↑ | Wino.<br>acc ↑ | ARC-e<br>acc ↑ | ARC-c<br>acc_n ↑ | Avg.        |" (Table 2, Section 4.3). |
| Hella. | Classification | Hella. | Natural language text | "| Model               | LMB.<br>ppl↓          | LMB.<br>acc ↑ | PIQA<br>acc ↑ | Hella.<br>acc_n ↑ | Wino.<br>acc ↑ | ARC-e<br>acc ↑ | ARC-c<br>acc_n ↑ | Avg.        |" (Table 2, Section 4.3). |
| Wino. | Classification | Wino. | Natural language text | "| Model               | LMB.<br>ppl↓          | LMB.<br>acc ↑ | PIQA<br>acc ↑ | Hella.<br>acc_n ↑ | Wino.<br>acc ↑ | ARC-e<br>acc ↑ | ARC-c<br>acc_n ↑ | Avg.        |" (Table 2, Section 4.3). |
| ARC-e | Classification | ARC-e | Natural language text | "| Model               | LMB.<br>ppl↓          | LMB.<br>acc ↑ | PIQA<br>acc ↑ | Hella.<br>acc_n ↑ | Wino.<br>acc ↑ | ARC-e<br>acc ↑ | ARC-c<br>acc_n ↑ | Avg.        |" (Table 2, Section 4.3). |
| ARC-c | Classification | ARC-c | Natural language text | "| Model               | LMB.<br>ppl↓          | LMB.<br>acc ↑ | PIQA<br>acc ↑ | Hella.<br>acc_n ↑ | Wino.<br>acc ↑ | ARC-e<br>acc ↑ | ARC-c<br>acc_n ↑ | Avg.        |" (Table 2, Section 4.3). |

## 4. Domain and Modality Scope

Evaluation scope: Multiple domains within the same modality (text/sequence). Evidence: "In the following section we test our proposed model on synthetic and real-world language modeling tasks." (Section 4 Experiments).

Multiple modalities: Not indicated; only sequence/text tasks are stated. Evidence: "synthetic and real-world language modeling tasks" (Section 4 Experiments).

Domain generalization or cross-domain transfer claims: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Multi-Query Associative Recall (MQAR) | Not specified. | Not specified. | Not specified. | "We evaluate GLA + *Selective RoPE* on Multi-Query Associative Recall" (Section 4.2). |
| MAD - Compress | Not specified. | Not specified. | Not specified. | "We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2). |
| MAD - Fuzzy Recall | Not specified. | Not specified. | Not specified. | "We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2). |
| MAD - In-Context Recall | Not specified. | Not specified. | Not specified. | "We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2). |
| MAD - Memorize | Not specified. | Not specified. | Not specified. | "We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2). |
| MAD - Noisy Recall | Not specified. | Not specified. | Not specified. | "We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2). |
| MAD - Selective Copy | Not specified. | Not specified. | Not specified. | "We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2). |
| String copying | Not specified. | Not specified. | Not specified. | "We also evaluate string copying following Jelassi et al. (2024)." (Section 4.2). |
| State tracking on permutation composition ($S_2$) | Not specified. | Not specified. | Not specified. | "State Tracking. A common way to evaluate the expressivity of a model is *state tracking* on permutation composition (Liu et al., 2023)." (Section 4.2). |
| State tracking on permutation composition ($A_3$) | Not specified. | Not specified. | Not specified. | "State Tracking. A common way to evaluate the expressivity of a model is *state tracking* on permutation composition (Liu et al., 2023)." (Section 4.2). |
| Language modeling (pretraining) | Not specified. | Not specified. | Not specified. | "For our language modeling experiments we train 370M parameter versions of GLA (Yang et al., 2024a), Gated DeltaNet (Yang et al., 2025a), and the Forgetting Transformer (FoX) (Lin et al., 2025)" (Section 4.3). |
| Lambada (LMB.) | Yes (shared pretrained model for zero-shot eval). | No (zero-shot). | Not specified. | "The best models are then evaluated on downstream tasks from lm-eval-harness (Gao et al., 2024)." and "We follow the default zero-shot evaluation setup in lm-eval-harness, using its standard prompting" (Section 4.3). |
| PIQA | Yes (shared pretrained model for zero-shot eval). | No (zero-shot). | Not specified. | "The best models are then evaluated on downstream tasks from lm-eval-harness (Gao et al., 2024)." and "We follow the default zero-shot evaluation setup in lm-eval-harness, using its standard prompting" (Section 4.3). |
| Hella. | Yes (shared pretrained model for zero-shot eval). | No (zero-shot). | Not specified. | "The best models are then evaluated on downstream tasks from lm-eval-harness (Gao et al., 2024)." and "We follow the default zero-shot evaluation setup in lm-eval-harness, using its standard prompting" (Section 4.3). |
| Wino. | Yes (shared pretrained model for zero-shot eval). | No (zero-shot). | Not specified. | "The best models are then evaluated on downstream tasks from lm-eval-harness (Gao et al., 2024)." and "We follow the default zero-shot evaluation setup in lm-eval-harness, using its standard prompting" (Section 4.3). |
| ARC-e | Yes (shared pretrained model for zero-shot eval). | No (zero-shot). | Not specified. | "The best models are then evaluated on downstream tasks from lm-eval-harness (Gao et al., 2024)." and "We follow the default zero-shot evaluation setup in lm-eval-harness, using its standard prompting" (Section 4.3). |
| ARC-c | Yes (shared pretrained model for zero-shot eval). | No (zero-shot). | Not specified. | "The best models are then evaluated on downstream tasks from lm-eval-harness (Gao et al., 2024)." and "We follow the default zero-shot evaluation setup in lm-eval-harness, using its standard prompting" (Section 4.3). |

## 6. Input and Representation Constraints

Sequence length and dimensionality: "transforms a sequence of L inputs  $(x_t)_{t=1}^L$  into the sequence of outputs  $(o_t)_{t=1}^L$ , with  $x_t, s_t, o_t \in \mathbb{R}^d$" (Section 2 Background).

RoPE dimensional structure: "For queries and keys  $q_t, k_\tau \in \mathbb{R}^2$ , *RoPE* applies relative positional encoding" and "For d-dimensional queries and keys,  $\mathbf{q}_t, \mathbf{k}_{\tau}$  are split into d/2 vectors  $\in \mathbb{R}^2$" (Section 2 Background).

Context length/tokenization: "of FineWeb (Penedo et al., 2024) at a context length of 4096 and use the Mistral 7B tokenizer (Jiang et al., 2023) with a vocabulary size of 32 000." (Section 4.3 Language Modeling).

Fixed patch size/resolution: Not specified.

Padding/resizing requirements: Not specified.

## 7. Context Window and Attention Structure

Maximum sequence length: "context length of 4096" (Section 4.3 Language Modeling).

Fixed or variable length: "a sequence of L inputs  $(x_t)_{t=1}^L$" (Section 2 Background) alongside a fixed training context length of 4096 for LM (Section 4.3 Language Modeling).

Attention type: Global causal attention over past tokens. Evidence: "the ability of every token to attend to all past tokens without decay" (Introduction).

Mechanisms to manage computational cost: "sub-quadratic sequence models (modern recurrent architectures) that run in *linear* time and require only *constant* memory per step at inference" (Introduction).

State management for long sequences: "to manage the finite sized hidden state better when processing long sequences, (2) was enhanced with a *forget gate*,  $A_t$" (Section 2 Background).

## 8. Positional Encoding (Critical Section)

Mechanism: "Rotary Position Embeddings (*RoPE*) are used to add relative positional information through rotations of the query-key pairs" and "we introduce Selective RoPE, an input-dependent rotary embedding mechanism, that generalizes RoPE" (Section 2 Background; Abstract).

Where applied: "applying a learned, input-dependent rotary position embedding to the queries and keys" and "Selective RoPE is easily incorporated into the query and keys of any gated linear transformer." (Introduction).

Fixed vs modified/ablated: "We ablate our architectural choices on the MAD dataset and language modeling experiments." and comparison rows include NoPE/RoPE/Selective RoPE in Table 1 (Section 4.1; Table 1).

## 9. Positional Encoding as a Variable

Core research variable: "We introduced *Selective RoPE*, an input-dependent rotary position embedding that generalizes RoPE from fixed to arbitrary, learnable rotations." (Conclusion).

Multiple positional encodings compared: "using *Selective RoPE* consistently improves performance over NoPE and RoPE" (Section 4.2 Synthetic Language Tasks).

Claim that PE choice is not critical or secondary: Not claimed.

## 10. Evidence of Constraint Masking

Model size(s): "we train 370M parameter versions of GLA (Yang et al., 2024a), Gated DeltaNet (Yang et al., 2025a), and the Forgetting Transformer (FoX) (Lin et al., 2025)" (Section 4.3 Language Modeling).

Dataset size(s): "All models are trained on 35B tokens" (Section 4.3 Language Modeling).

Attribution of gains: "equipping certain sequence models (namely, GLA, Gated DeltaNet, and FoX) with *Selective RoPE* improves recall-centric synthetic tasks and strengthens language modeling downstream performance." (Conclusion).

Scaling-based attribution: Not claimed.

## 11. Architectural Workarounds

Linear-time recurrent/linear attention to reduce quadratic cost: "sub-quadratic sequence models (modern recurrent architectures) that run in *linear* time and require only *constant* memory per step at inference" (Introduction).

Forget gate for managing finite state over long sequences: "to manage the finite sized hidden state better when processing long sequences, (2) was enhanced with a *forget gate*,  $A_t$" (Section 2 Background).

RoPE trick to implement complex rotations efficiently in real domain: "The RoPE trick allows us to implement this complex parameterization by applying RoPE to queries and keys, effectively staying in the real domain" (Section 2 Background).

Phase gate/bias/weight norm additions: "We also add a learnable bias term, which is not dependent on relative token positions (Li et al., 2024). Finally, we place a weight norm (Kingma, 2016) on the input projection." (Section 4.1 Implementation).

## 12. Explicit Limitations and Non-Claims

Length extrapolation not studied: "we note that incorporating RoPE is notoriously detrimental to the length-extrapolation capabilities of sequence models (Li et al., 2024). In this paper, we do not investigate this aspect since we consider it to be out of the scope of our research." (Conclusion: Future work).

Need to study extra components: "further investigation of the effect of the extra components used in *Selective RoPE*, namely the bias term and the phase gate, can be a fruitful direction for future research." (Conclusion: Future work).

Forget gate dimensionality not settled: "we consider the impact of choosing a diagonal as opposed to a scalar forget gate to be an interesting question" (Conclusion: Future work).

Positional embedding variants left to future work: "we believe it to be important to also incorporate the progress on the positional embedding front into future work." (Conclusion: Future work).

Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Synthetic sequence tasks plus real-world language modeling; single text modality.
> - Task structure: Recall-heavy synthetic benchmarks (MQAR, MAD, copying, state tracking) and multiple NLP benchmarks from lm-eval-harness.
> - Representation rigidity: Fixed tokenization (Mistral 7B, vocab size 32 000) and fixed LM context length (4096); fixed d-dimensional token representations.
> - Model sharing vs specialization: Zero-shot evaluation on lm-eval-harness implies shared pretrained weights; synthetic task training regime not specified.
> - Role of positional encoding: Central variable, with Selective RoPE compared against NoPE/RoPE and component ablations.

### 14. Final Classification

Classification: **Multi-task, multi-domain (constrained)**.

Justification: The paper explicitly evaluates on "synthetic and real-world language modeling tasks" (Section 4), spanning multiple distinct tasks across synthetic and natural-language domains within a single modality. The setup remains constrained to sequence modeling with fixed tokenizer/context settings and focuses on positional embedding variants rather than cross-domain transfer claims (Section 4.3; Abstract).
