## 1. Basic Metadata

Title: "Transformers Can Do Arithmetic with the Right Embeddings" (Title)
Authors: "Sean McLeish<sup>1\*</sup>, Arpit Bansal<sup>1\*</sup>, Alex Stein<sup>1</sup>, Neel Jain<sup>1</sup>, John Kirchenbauer<sup>1</sup>, Brian R. Bartoldson<sup>2</sup>, Bhavya Kailkhura<sup>2</sup>, Abhinav Bhatele<sup>1</sup>, Jonas Geiping<sup>3</sup>, Avi Schwarzschild<sup>4</sup>, Tom Goldstein<sup>1</sup>" (Title)
Year: Year not specified.
Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper claims it "mend[s] this problem by adding an embedding to each digit that encodes its position relative to the start of the number," aiming to fix transformers' arithmetic performance. (Abstract)

## 3. Tasks Evaluated

| Task name | Task type | Dataset(s) used | Domain | Evidence |
| --- | --- | --- | --- | --- |
| Addition | Generation; Reasoning / relational | "For this study all training sets have 20 million samples and i = j, hence we can use one number to define the dataset i, where i is the maximum length of either operand." (Section 3) | "inputs are formatted least significant digit first, e.g. 98282 + 3859172 = 2787472." (Section 3) | "We train decoder-only causal language models to solve addition problems." (Section 3) |
| Subtraction (joint with addition) | Generation; Reasoning / relational | "We train models on a dataset made up of an even mix of addition and subtraction samples." (Section 4.1) | "We train models on a dataset made up of an even mix of addition and subtraction samples." (Section 4.1) | "We train models on a dataset made up of an even mix of addition and subtraction samples." (Section 4.1) |
| Multiplication | Generation; Reasoning / relational | "We implement the multiplication datasets for both training and testing the exact same manor as for addition, only changing the operation used to calculate the answer." (Appendix A.2) | "multiplication of natural numbers, where the length of the output may be the sum of the lengths of the operands." (Section 4.2) | "We now study a harder task, multiplication of natural numbers, where the length of the output may be the sum of the lengths of the operands." (Section 4.2) |
| Sorting arrays of numbers | Generation; Reasoning / relational | "We train with arrays of up to 10 numbers each having up to 10 digits and then evaluate with arrays of up to 30 numbers each having up to 30 digits." (Section 4.3) | "Given a list of reversed integers indexed by characters, output the characters in ascending order." (Appendix A.2) | "we now analyze the task of sorting arrays of multiple variable length numbers" (Section 4.3) |
| Bitwise OR (binary vectors) | Generation; Reasoning / relational | "For training, we exhaustively sample the space of all vectors of sizes less than or equal to the predefined maximum input vector size." (Appendix A.2) | "The input for this problem is two binary vectors, the longer input vector is all zeros and the shorter input contains a one. The output should be the length of the longer vector with the one in the same position as in the shorter vector." (Appendix A.2) | "We train standard transformer, standard transformer with input injection and looped transformer models on the position wise or task, on a dataset where the maximum length of either input vector is twenty." (Appendix A.3) |

## 4. Domain and Modality Scope

Single domain? Yes; "In this paper, we study mathematical reasoning tasks including addition, multiplication, and sorting to evaluate these capabilities in a controlled setting." (Section 5)
Multiple domains within the same modality? Not claimed; "We use a character level tokenizer for all experiments and greedy decoding in all testing." (Appendix A.8)
Multiple modalities? Not specified; "We use a character level tokenizer for all experiments and greedy decoding in all testing." (Appendix A.8)
Domain generalization or cross-domain transfer? Not claimed; "we do not actually explore any natural language tasks." (Section 5 Limitations)

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Addition | N/A (single-task training stated) | Not specified | Not specified | "We train decoder-only causal language models to solve addition problems." (Section 3) |
| Subtraction (joint with addition) | Yes (joint training with addition) | Not specified | Not specified | "We train models on a dataset made up of an even mix of addition and subtraction samples." (Section 4.1) |
| Multiplication | N/A (single-task training stated) | Not specified | Not specified | "We now study a harder task, multiplication of natural numbers, where the length of the output may be the sum of the lengths of the operands." (Section 4.2) |
| Sorting | N/A (single-task training stated) | Not specified | Not specified | "We train with arrays of up to 10 numbers each having up to 10 digits and then evaluate with arrays of up to 30 numbers each having up to 30 digits." (Section 4.3) |
| Bitwise OR | N/A (single-task training stated) | Not specified | Not specified | "We train standard transformer, standard transformer with input injection and looped transformer models on the position wise or task, on a dataset where the maximum length of either input vector is twenty." (Appendix A.3) |

## 6. Input and Representation Constraints

| Constraint | Evidence |
| --- | --- |
| Least significant digit first formatting for arithmetic inputs | "inputs are formatted least significant digit first, e.g. 98282 + 3859172 = 2787472." (Section 3) |
| No padding between digits; no zero padding to equalize lengths | "Unlike prior work, we do not add any padding between digits [Shen et al., 2023] and do not pad any numbers with zeros, neither in the case of carry digits [Zhou et al., 2024], nor to make all operands the same length [Shen et al., 2023]." (Section 3) |
| Variable operand lengths up to maximum i and j | "We train on all combinations of operand lengths less than or equal to i and j where i and j are the maximum lengths of the first and second operands, respectively." (Section 3) |
| Character-level tokenization | "We use a character level tokenizer for all experiments and greedy decoding in all testing." (Appendix A.8) |
| Sorting inputs are reversed integers with bounded array and digit lengths | "Given a list of reversed integers indexed by characters, output the characters in ascending order." (Appendix A.2) / "We train with arrays of up to 10 numbers each having up to 10 digits and then evaluate with arrays of up to 30 numbers each having up to 30 digits." (Section 4.3) |
| Bitwise OR vector length bound | "We train standard transformer, standard transformer with input injection and looped transformer models on the position wise or task, on a dataset where the maximum length of either input vector is twenty." (Appendix A.3) |

## 7. Context Window and Attention Structure

Maximum sequence length: Not specified; only "capped by the context length" (Section 5)
Fixed or variable sequence length: Variable; "We train on all combinations of operand lengths less than or equal to i and j where i and j are the maximum lengths of the first and second operands, respectively." (Section 3)
Attention type: Not specified; models are "decoder-only causal language models" and a "standard autoregressive transformer model where multiple decoder layers are stacked in a feedforward manner." (Section 3)
Compute-cost mechanisms (windowing/pooling/pruning): Not specified.

## 8. Positional Encoding (Critical Section)

| Positional encoding | Type | Where applied | Evidence |
| --- | --- | --- | --- |
| Abacus Embeddings | Absolute (learned) | Input embeddings for digit positions | "Our *Abacus Embeddings* are simple learned positional embeddings that are used to encode positions within each span of numerical tokens." (Introduction) / "We apply the same positional embedding to all digits of the same significance" (Section 3.1) / "As Abacus Embeddings are a variant of absolute positional embeddings" (Section 3.1) |
| APE (absolute positional embeddings) | Absolute | Input only | "Absolute positional embeddings (APE) are learned embeddings that are added to token embeddings before the first layer of the transformer [Vaswani et al., 2017]." (Related Work: Positional Embeddings) |
| FIRE | Relative / bias-based | Attention mechanism | "FIRE embeddings are additive embeddings in the attention mechanism:  $A_{RPE}(X) = XW_Q(XW_K)^T + B$" (Appendix A.1.1) |
| NoPE | Implicit / none | None | "No positional embeddings (NoPE) can achieve good length generalization performance for small algorithmic tasks and even outperform some specialized embeddings." (Related Work: Positional Embeddings) |
| RoPE | RoPE | Not specified (rotary) | "Rotary Positional Embeddings (RoPE) [Su et al., 2024] are commonly used in state-of-the-art open source transformers" (Related Work: Positional Embeddings) |

Positional encodings are varied across experiments: "In this work, we focus on NoPE and FIRE embeddings since these are the best performers for addition in reversed format among existing embeddings [Zhou et al., 2024]." (Related Work: Positional Embeddings) and "Combining Abacus Embeddings with FIRE or RoPE embeddings improves out of distribution accuracy for addition, over the baseline models without Abacus Embeddings." (Section 4.4)

## 9. Positional Encoding as a Variable

Core research variable? Yes: "We propose a new positional embedding called *Abacus Embeddings* to better capture the significance of each digit, which leads to near-perfect in-distribution generalization." (Introduction)
Multiple positional encodings compared? Yes: "varying the architecture and embeddings." (Figure 3 caption) and "Table 1: Exact match accuracy for sorting with various positional embeddings." (Section 4.3)
Claim PE not critical or secondary? Not stated.

## 10. Evidence of Constraint Masking

- Dataset size: "For this study all training sets have 20 million samples and i = j, hence we can use one number to define the dataset i, where i is the maximum length of either operand." (Section 3)
- Compute budget: "we use a language model cramming setup [Geiping and Goldstein, 2023] and limit each training run to 8 exaFLOP of compute (a single Nvidia RTXA4000 GPU for 24 hours); for multiplication results we allow 64 exaFLOP (eight Nvidia RTXA4000 GPUs for 24 hours)." (Section 3)
- Model sizes: "Table 3: Number of parameters, to the nearest million, in a model with Abacus Embeddings and input injection." (Appendix A.8) and "| 16                        | 1           | 122                   |" (Table 3)
- Data scale effects: "Across Figure 9, we see that increasing the size of the operands in the training set allows for better generalization above one hundred digits for all models." (Appendix A.4)
- Performance gains attributed to architectural/embedding choices: "Combining Abacus Embeddings and standard positional embeddings, we observe dramatic improvements in accuracy" and "We show that when we combine Abacus Embeddings with input injection and looped transformers performance further improves, increasing from 92.9% to 99.1% in out of distribution accuracy, an 87% reduction in error compared to using the embeddings with standard architectures alone." (Introduction)

## 11. Architectural Workarounds

| Technique | Purpose | Evidence |
| --- | --- | --- |
| Input injection (skip connections) | Improve generalization on arithmetic tasks | "we enhance this standard transformer model by incorporating *input injection*, where the embedded inputs are added to the input of each decoder layer [Ma et al., 2022, Bansal et al., 2022, Anil et al., 2022a]." (Section 3) |
| Looped transformers / recurrence (weight sharing) | Improve generalization with recurrent computation | "looped transformer architectures, which contain recurrent layers in which the same parameters are re-used multiple times" (Introduction) / "Looped transformer (LT): Weight tied decoder layers, with input injection and progressive loss." (Figure 3 caption) |
| Progressive loss (varying recurrences) | Encourage generalization with variable recurrences | "refered to as *progressive loss* computation [Bansal et al., 2022]. This loss function is a convex combination of the loss values from two forward passes" (Section 3.2) |
| Loss masking before answer digits | Training trick to focus on outputs | "During training, we mask the input question and only compute loss on the answer digits." (Section 3) |

## 12. Explicit Limitations and Non-Claims

- "There are some intrinsic limitations that accompany any study involving language model training from scratch under compute constraints." (Section 5 Limitations)
- "although we show the compatibility of Abacus Embeddings with FIRE and RoPE embeddings, we do not actually explore any natural language tasks." (Section 5 Limitations)
- "In the future, a larger scale study including natural language would be needed to understand further how Abacus Embeddings would perform on heterogeneous tasks comprising both numerical and natural language inputs." (Section 5 Limitations)

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Synthetic algorithmic/numeric sequences; no natural language evaluation.
> - Task structure: Algorithmic sequence prediction tasks (addition, subtraction, multiplication, sorting, bitwise OR) with exact-match accuracy.
> - Representation rigidity: Reversed digit format, character-level tokens, no padding, fixed maximum lengths per dataset.
> - Model sharing vs specialization: Mostly single-task training; one joint addition+subtraction setup.
> - Role of positional encoding: Central experimental variable with multiple PE variants and combinations.

### 14. Final Classification

Final classification: **Multi-task, single-domain**. The study evaluates multiple algorithmic tasks in the same numeric setting: "In this paper, we study mathematical reasoning tasks including addition, multiplication, and sorting to evaluate these capabilities in a controlled setting." (Section 5) It also includes joint training on "an even mix of addition and subtraction samples." (Section 4.1) The scope explicitly excludes cross-domain evaluation: "we do not actually explore any natural language tasks." (Section 5 Limitations)
