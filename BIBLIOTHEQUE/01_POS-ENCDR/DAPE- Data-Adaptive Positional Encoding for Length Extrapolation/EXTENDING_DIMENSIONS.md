## 1. Basic Metadata

- Title: DAPE: Data-Adaptive Positional Encoding for Length Extrapolation
- Authors: Chuanyang Zheng; Yihang Gao; Han Shi; Minbin Huang; Jingyao Li; Jing Xiong; Xiaozhe Ren; Michael Ng; Xin Jiang; Zhenguo Li; Yu Li
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper proposes a data-adaptive positional encoding method for transformers to improve length extrapolation and long-context performance ("we propose a Data-Adaptive Positional Encoding (DAPE) method, which dynamically and semantically adjusts based on input context and learned fixed priors." - Abstract).

---

## 3. Tasks Evaluated

- Task name: Language modeling (perplexity evaluation) on Arxiv
  - Task type: Generation
  - Dataset(s) used: Arxiv
  - Domain: natural language text
  - Evidence: "Our analysis involves training language models on the Arxiv and Books3 datasets" and "We start our evaluation by comparing the last 256 tokens' zero-shot perplexity across different input lengths." (Section 4 Experiment)

- Task name: Language modeling (perplexity evaluation) on Books3
  - Task type: Generation
  - Dataset(s) used: Books3
  - Domain: natural language text
  - Evidence: "Our analysis involves training language models on the Arxiv and Books3 datasets" and "We start our evaluation by comparing the last 256 tokens' zero-shot perplexity across different input lengths." (Section 4 Experiment)

- Task name: EVEN PAIRS (CHE)
  - Task type: Classification; Reasoning / relational
  - Dataset(s) used: Chomsky Hierarchy Evaluation Benchmark (CHE)
  - Domain: synthetic formal language strings
  - Evidence: "|         | EVEN PAIRS                  | aabba                                                                                           | True                  |" (Appendix D, Table 4)

- Task name: MODULAR ARITHMETIC (SIMPLE) (CHE)
  - Task type: Generation; Reasoning / relational
  - Dataset(s) used: Chomsky Hierarchy Evaluation Benchmark (CHE)
  - Domain: synthetic formal language strings
  - Evidence: "| Dagular | MODULAR ARITHMETIC (SIMPLE) | 1 + 2 - 4                                                                                       | 4                     |" (Appendix D, Table 4)

- Task name: PARITY CHECK††† (CHE)
  - Task type: Classification; Reasoning / relational
  - Dataset(s) used: Chomsky Hierarchy Evaluation Benchmark (CHE)
  - Domain: synthetic formal language strings
  - Evidence: "| Regular | PARITY CHECK†††             | aaabba                                                                                          | True                  |" (Appendix D, Table 4)

- Task name: CYCLE NAVIGATION††† (CHE)
  - Task type: Generation; Reasoning / relational
  - Dataset(s) used: Chomsky Hierarchy Evaluation Benchmark (CHE)
  - Domain: synthetic formal language strings
  - Evidence: "|         | CYCLE NAVIGATION†††         | 011210                                                                                          | 2                     |" (Appendix D, Table 4)

- Task name: STACK MANIPULATION (CHE)
  - Task type: Generation; Reasoning / relational
  - Dataset(s) used: Chomsky Hierarchy Evaluation Benchmark (CHE)
  - Domain: synthetic formal language strings
  - Evidence: "|         | STACK MANIPULATION          | abbaa POP PUSH a POP                                                                            | abba                  |" (Appendix D, Table 4)

- Task name: REVERSE STRING (CHE)
  - Task type: Generation; Reasoning / relational
  - Dataset(s) used: Chomsky Hierarchy Evaluation Benchmark (CHE)
  - Domain: synthetic formal language strings
  - Evidence: "| DCE     | REVERSE STRING              | aabba                                                                                           | abbaa                 |" (Appendix D, Table 4)

- Task name: MODULAR ARITHMETIC (CHE)
  - Task type: Generation; Reasoning / relational
  - Dataset(s) used: Chomsky Hierarchy Evaluation Benchmark (CHE)
  - Domain: synthetic formal language strings
  - Evidence: "| DCF     | MODULAR ARITHMETIC          | $-(1-2)\cdot(4-3\cdot(-2))$                                                                     | 0                     |" (Appendix D, Table 4)

- Task name: SOLVE EQUATION (CHE)
  - Task type: Generation; Reasoning / relational
  - Dataset(s) used: Chomsky Hierarchy Evaluation Benchmark (CHE)
  - Domain: synthetic formal language strings
  - Evidence: "|         | SOLVE EQUATION              | $ \begin{array}{l} -(1-2) \cdot (4-3 \cdot (-2)) \\ -(x-2) \cdot (4-3 \cdot (-2)) \end{array} $ | 1                     |" (Appendix D, Table 4)

- Task name: DUPLICATE STRING (CHE)
  - Task type: Generation; Reasoning / relational
  - Dataset(s) used: Chomsky Hierarchy Evaluation Benchmark (CHE)
  - Domain: synthetic formal language strings
  - Evidence: "|         | DUPLICATE STRING            | abaab                                                                                           | abaababaab            |" (Appendix D, Table 4)

- Task name: MISSING DUPLICATE (CHE)
  - Task type: Generation; Reasoning / relational
  - Dataset(s) used: Chomsky Hierarchy Evaluation Benchmark (CHE)
  - Domain: synthetic formal language strings
  - Evidence: "|         | MISSING DUPLICATE           | 10011021                                                                                        | 0                     |" (Appendix D, Table 4)

- Task name: Odds First (CHE)
  - Task type: Generation; Reasoning / relational
  - Dataset(s) used: Chomsky Hierarchy Evaluation Benchmark (CHE)
  - Domain: synthetic formal language strings
  - Evidence: "|         | Odds First                  | aaabaa                                                                                          | aaaaba                |" (Appendix D, Table 4)

- Task name: BINARY ADDITION (CHE)
  - Task type: Generation; Reasoning / relational
  - Dataset(s) used: Chomsky Hierarchy Evaluation Benchmark (CHE)
  - Domain: synthetic formal language strings
  - Evidence: "| CS      | BINARY ADDITION             | 10010 + 101                                                                                     | 10111                 |" (Appendix D, Table 4)

- Task name: COMPUTE SQRT (CHE)
  - Task type: Generation; Reasoning / relational
  - Dataset(s) used: Chomsky Hierarchy Evaluation Benchmark (CHE)
  - Domain: synthetic formal language strings
  - Evidence: "|         | COMPUTE SQRT                | 100010                                                                                          | 110                   |" (Appendix D, Table 4)

- Task name: BUCKET SORT††† (CHE)
  - Task type: Generation; Reasoning / relational
  - Dataset(s) used: Chomsky Hierarchy Evaluation Benchmark (CHE)
  - Domain: synthetic formal language strings
  - Evidence: "|         | BUCKET SORT†††              | 421302214                                                                                       | 011222344             |" (Appendix D, Table 4)

---

## 4. Domain and Modality Scope

- Evaluation scope: Multiple domains within the same modality (text). Evidence: "Our analysis involves training language models on the Arxiv and Books3 datasets" and "we also evaluated DAPE on downstream Chomsky Hierarchy Evaluation Benchmark (CHE) [21]" (Section 4 Experiment; Section 4.7 Experiments on CHE Benchmark).
- Domain generalization or cross-domain transfer claims: Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Language modeling (Arxiv) | Not specified. | Not specified. | Not specified. | "Our analysis involves training language models on the Arxiv and Books3 datasets" (Section 4 Experiment) |
| Language modeling (Books3) | Not specified. | Not specified. | Not specified. | "Our analysis involves training language models on the Arxiv and Books3 datasets" (Section 4 Experiment) |
| EVEN PAIRS (CHE) | Not specified. | Not specified. | Not specified. | "we conduct evaluations of our DAPE on a suite of tasks derived from the domain of formal language recognition" (Appendix D) |
| MODULAR ARITHMETIC (SIMPLE) (CHE) | Not specified. | Not specified. | Not specified. | "we conduct evaluations of our DAPE on a suite of tasks derived from the domain of formal language recognition" (Appendix D) |
| PARITY CHECK††† (CHE) | Not specified. | Not specified. | Not specified. | "we conduct evaluations of our DAPE on a suite of tasks derived from the domain of formal language recognition" (Appendix D) |
| CYCLE NAVIGATION††† (CHE) | Not specified. | Not specified. | Not specified. | "we conduct evaluations of our DAPE on a suite of tasks derived from the domain of formal language recognition" (Appendix D) |
| STACK MANIPULATION (CHE) | Not specified. | Not specified. | Not specified. | "we conduct evaluations of our DAPE on a suite of tasks derived from the domain of formal language recognition" (Appendix D) |
| REVERSE STRING (CHE) | Not specified. | Not specified. | Not specified. | "we conduct evaluations of our DAPE on a suite of tasks derived from the domain of formal language recognition" (Appendix D) |
| MODULAR ARITHMETIC (CHE) | Not specified. | Not specified. | Not specified. | "we conduct evaluations of our DAPE on a suite of tasks derived from the domain of formal language recognition" (Appendix D) |
| SOLVE EQUATION (CHE) | Not specified. | Not specified. | Not specified. | "we conduct evaluations of our DAPE on a suite of tasks derived from the domain of formal language recognition" (Appendix D) |
| DUPLICATE STRING (CHE) | Not specified. | Not specified. | Not specified. | "we conduct evaluations of our DAPE on a suite of tasks derived from the domain of formal language recognition" (Appendix D) |
| MISSING DUPLICATE (CHE) | Not specified. | Not specified. | Not specified. | "we conduct evaluations of our DAPE on a suite of tasks derived from the domain of formal language recognition" (Appendix D) |
| Odds First (CHE) | Not specified. | Not specified. | Not specified. | "we conduct evaluations of our DAPE on a suite of tasks derived from the domain of formal language recognition" (Appendix D) |
| BINARY ADDITION (CHE) | Not specified. | Not specified. | Not specified. | "we conduct evaluations of our DAPE on a suite of tasks derived from the domain of formal language recognition" (Appendix D) |
| COMPUTE SQRT (CHE) | Not specified. | Not specified. | Not specified. | "we conduct evaluations of our DAPE on a suite of tasks derived from the domain of formal language recognition" (Appendix D) |
| BUCKET SORT††† (CHE) | Not specified. | Not specified. | Not specified. | "we conduct evaluations of our DAPE on a suite of tasks derived from the domain of formal language recognition" (Appendix D) |

---

## 6. Input and Representation Constraints

- Fixed/variable sequence length (language modeling): "training lengths of 128, 512, and 1024" and evaluation at longer lengths, e.g., "evaluation sequence length 8192" (Section 4 Experiment settings; Abstract).
- Perplexity evaluation uses fixed output segment: "we initially utilize the model to process the entire input sentence and subsequently select the final 256 tokens for perplexity computation." (Section 5 Evaluation Protocol)
- CHE task input construction: "In scenarios that necessitate a multi-token output sequence y, such as the task of string duplication, we extend the input sequence by appending |y| placeholder tokens." (Appendix D)
- CHE sequence-length constraints: "Training is conducted on sequences whose lengths are uniformly distributed, sampled from U(1,N), with N set to 40. Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500." (Appendix D)
- Fixed patch size / fixed number of tokens / fixed 2D dimensionality / padding or resizing requirements: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length: "evaluation sequence length 8192" (Abstract) and CHE evaluation length up to "M equals 500" (Appendix D).
- Fixed vs variable length: fixed training lengths for LM ("training lengths of 128, 512, and 1024") and variable evaluation lengths ("evaluation sequence length 8192"), plus CHE variable evaluation lengths ("Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500.") (Section 4 Experiment settings; Abstract; Appendix D).
- Attention type: Global (full softmax attention). Evidence: "The attention block was originally designed by applying softmax to the key-query multiplication, which requires quadratic computational cost." (Section 1 Introduction) and "$$\mathbf{A}_{\text{RPE}}(\mathbf{X}) = \mathbf{X} \mathbf{W}_{Q} (\mathbf{X} \mathbf{W}_{K})^{\top} + \mathbf{B}, \tag{1}$$" (Section 3.1).
- Mechanisms to manage computational cost: No windowing or sparse attention is introduced; DAPE adds an MLP-based bias with cost "the additional computational costs are $\mathcal{O}\left(hN^2D_{\text{DAPE}}\right)$." and notes that if "$D_{\text{DAPE}} \ll d$ , the incremental computational cost introduced by DAPE is not significant." (Section 3.2 Multi-head DAPE).

---

## 8. Positional Encoding (Critical Section)

- Mechanism: Relative, bias-based (additive) positional encoding with data-adaptive MLP adjustments. Evidence: "For most additive RPE methods, the computation of pre-softmax attention logits can be unified under the following formula:" and "$$\mathbf{A}_{\text{RPE}}(\mathbf{X}) = \mathbf{X} \mathbf{W}_{Q} (\mathbf{X} \mathbf{W}_{K})^{\top} + \mathbf{B}, \tag{1}$$" (Section 3.1) and "we propose a Data-Adaptive Positional Encoding (DAPE) method, which dynamically and semantically adjusts based on input context and learned fixed priors." (Abstract).
- Where it is applied: Pre-softmax attention logits / attention bias. Evidence: "Thus, the pre-softmax attention logit incorporated with DAPE is formulated as" and "$$\mathbf{A}_{\text{DAPE}}(\mathbf{X}) = \mathbf{X} \mathbf{W}_{Q}(\mathbf{X} \mathbf{W}_{K})^{\top} + f(\mathbf{X} \mathbf{W}_{Q}(\mathbf{X} \mathbf{W}_{K})^{\top}, \mathbf{B}).$$" (Section 3.2).
- Fixed vs modified/compared: Multiple positional encodings are compared and DAPE variants use different bias matrices. Evidence: "We evaluate the proposed DAPE against a range of established baselines, including NoPE [33], RoPE [62], YaRN [51], Randomized RoPE [57, 31], T5's Bias [56], Alibi [52], Kerple [13], and FIRE [41]." (Section 4 Experiment) and "DAPE-Alibi, DAPE-Kerple, and DAPE-FIRE" (Section 4.1).

---

## 9. Positional Encoding as a Variable

- Core research variable: Yes. Evidence: "Positional encoding plays a crucial role in transformers, significantly impacting model performance and length generalization." and "we propose a Data-Adaptive Positional Encoding (DAPE) method" (Abstract).
- Multiple positional encodings compared: Yes. Evidence: "We evaluate the proposed DAPE against a range of established baselines, including NoPE [33], RoPE [62], YaRN [51], Randomized RoPE [57, 31], T5's Bias [56], Alibi [52], Kerple [13], and FIRE [41]." (Section 4 Experiment).
- Claim that PE choice is not critical or secondary: Not stated.

---

## 10. Evidence of Constraint Masking

- Model sizes: "model size 125M decoder-only Transformers" and "we evaluate the performance of larger model size 350M" (Section 4 Experiment settings); "whatever the model size is 2.7B or 6.7B." (Appendix H).
- Dataset sizes: Not specified.
- Attribution of gains: Performance gains are attributed to DAPE/semantic adaptivity rather than scaling data. Evidence: "DAPE consistently demonstrates an improvement in performance metrics" and "mainly due to the adoption of semantically adaptive PEs." (Section 4.2 The Effect of Model Size).
- Training tricks / scaling: Randomized positional encoding is used as a baseline setup: "For RoPE, the randomized positional encoding [57, 31] is applied to enhance the model performance." (Section 4 Experiment).

---

## 11. Architectural Workarounds

- Residual connection for positional information to improve optimization stability: "we introduce the residual connection for positional information." (Section 3.2).
- Multi-head DAPE mechanism to incorporate semantic and positional information across heads: "the DAPE in a multi-head setup processes the key-query similarities and bias matrices from all heads." (Section 3.2).
- No windowed or hierarchical attention is introduced; attention remains quadratic: "The attention block was originally designed by applying softmax to the key-query multiplication, which requires quadratic computational cost." (Section 1 Introduction).

---

## 12. Explicit Limitations and Non-Claims

- Limitations/time cost noted: "Practical additional time cost." and "The additional training ratio will gradually decrease with a larger model size, compared to baseline Kerple." (Section 4.5 The Time Cost).
- Explicit limitation statement in checklist: "We discuss the limitation in the Time Cost part (Section 4.5) of Experiment." (NeurIPS Paper Checklist).
- Explicit non-claims (open-world learning, unrestrained multi-task learning, meta-learning): Not specified.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Text-only evaluation across natural language corpora (Arxiv, Books3) and synthetic formal-language tasks (CHE).
> – Task structure: Language modeling (perplexity) plus multiple algorithmic/formal-language tasks with fixed evaluation protocols.
> – Representation rigidity: Sequence-length constraints are explicit (fixed training lengths; variable evaluation lengths; CHE length ranges).
> – Model sharing vs specialization: Training/evaluation per dataset/task is described, but weight sharing across tasks is not specified.
> – Role of positional encoding: Central experimental variable with multiple PE baselines and DAPE variants.

---

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple tasks within the same text modality: language modeling on "the Arxiv and Books3 datasets" and formal-language tasks in the "Chomsky Hierarchy Evaluation Benchmark (CHE)." It does not claim cross-domain transfer beyond text, and all evaluations are within language/text sequences.
