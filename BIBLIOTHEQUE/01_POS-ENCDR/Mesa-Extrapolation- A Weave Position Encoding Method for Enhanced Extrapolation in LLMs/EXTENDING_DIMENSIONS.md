## 1. Basic Metadata

- Title: "Mesa-Extrapolation: A Weave Position Encoding Method for Enhanced Extrapolation in LLMs" (Title block)
- Authors: "$Xin\ Ma^1,\ Yang\ Liu^{2,3},\ Jingjing\ Liu^2,\ Xiaoxu\ Ma^1$" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper claims to address LLM length extrapolation by introducing Mesa-Extrapolation, stating that "with meticulous weave position, PE can indeed be extended beyond effective range" and that it "utilizes a chunk-based triangular attention matrix and applies Stair PE to manage the final chunk." (Abstract)

## 3. Tasks Evaluated

Task: Passkey Retrieval (passkey dataset)
Task type: Other (retrieval / key finding)
Dataset(s): passkey dataset
Domain: text (synthetic)
Quotes: "We assess the accuracy of Mesa-Extrapolation using the generated passkey dataset. This dataset comprises samples of varying lengths, each storing a random password at a random position." (Section 5.1 Evaluation on Passkey Retrieval Tasks); "The LLM is required to find the correct password from the sample." (Appendix B.1 Passkey Retrieval Dataset)

Task: Language modeling (perplexity / NLL)
Task type: Generation
Dataset(s): Pile
Domain: text
Quotes: "We further assess the fluency of Mesa-Extrapolation utilizing the perplexity metric. Results evaluated on the Pile dataset are presented in Fig.4." (Section 5.2 Evaluation on Language Modeling); "The pile: An 800gb dataset of diverse text for language modeling." (References)

Task: Summarization
Task type: Generation
Dataset(s): GovReport
Domain: text
Quotes: "We conduct a summary task using the GovReport dataset and employ ROUGE [31] (ROUGE-1/2/L) as evaluation metrics." (Section 5.3 Evaluation on Summary of Tasks); "task is to generate a summary for texts of varying lengths, limited to 1000 tokens." (Appendix C.8 Generated Summary using Mesa-Extrapolation)

Task: LongEval Lines Task
Task type: Other (lines task; exact output format not specified)
Dataset(s): LongEval
Domain: long texts
Quotes: "We conduct additional testing on LongEval [21] lines task, a recently prominent evaluation task for long texts." (Appendix C.3 Evaluation on LongEval)

Task: LongBench Single-Document QA
Task type: Reasoning / relational
Dataset(s): qasper
Domain: text
Quotes: "We select LongBench [3] dataset and use 5 major categories of tasks, including Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion. Among them, each task selects a dataset, namely qasper, hotpotqa, samsum, passage-retrieval-en, and repobench-p." (Appendix C.4 Evaluation on LongBench)

Task: LongBench Multi-Document QA
Task type: Reasoning / relational
Dataset(s): hotpotqa
Domain: text
Quotes: "We select LongBench [3] dataset and use 5 major categories of tasks, including Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion. Among them, each task selects a dataset, namely qasper, hotpotqa, samsum, passage-retrieval-en, and repobench-p." (Appendix C.4 Evaluation on LongBench)

Task: LongBench Few-shot Learning
Task type: Other (few-shot learning evaluation)
Dataset(s): samsum
Domain: text
Quotes: "We select LongBench [3] dataset and use 5 major categories of tasks, including Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion. Among them, each task selects a dataset, namely qasper, hotpotqa, samsum, passage-retrieval-en, and repobench-p." (Appendix C.4 Evaluation on LongBench)

Task: LongBench Synthesis Tasks
Task type: Generation
Dataset(s): passage-retrieval-en
Domain: text
Quotes: "We select LongBench [3] dataset and use 5 major categories of tasks, including Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion. Among them, each task selects a dataset, namely qasper, hotpotqa, samsum, passage-retrieval-en, and repobench-p." (Appendix C.4 Evaluation on LongBench)

Task: LongBench Code Completion
Task type: Generation
Dataset(s): repobench-p
Domain: code
Quotes: "We select LongBench [3] dataset and use 5 major categories of tasks, including Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion. Among them, each task selects a dataset, namely qasper, hotpotqa, samsum, passage-retrieval-en, and repobench-p." (Appendix C.4 Evaluation on LongBench)

## 4. Domain and Modality Scope

- Evaluation scope: Multiple domains within the same modality (text), spanning QA and code tasks ("Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion") plus retrieval and summarization ("We assess the accuracy of Mesa-Extrapolation using the generated passkey dataset."; "We conduct a summary task using the GovReport dataset and employ ROUGE [31] (ROUGE-1/2/L) as evaluation metrics."). (Appendix C.4 Evaluation on LongBench; Section 5.1; Section 5.3)
- Multiple modalities: Not stated; evaluations are on LLM token sequences and long-text benchmarks. "We conduct additional testing on LongEval [21] lines task, a recently prominent evaluation task for long texts." (Appendix C.3 Evaluation on LongEval)
- Domain generalization / cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Passkey Retrieval | Not specified (plug-in; no fine-tuning) | No | Not mentioned | "Since our method is completely free plug-in and does not require fine-tuning" (Section 5 Experiments) |
| Language modeling (PPL/NLL) | Not specified (plug-in; no fine-tuning) | No | Not mentioned | "Since our method is completely free plug-in and does not require fine-tuning" (Section 5 Experiments) |
| Summarization (GovReport) | Not specified (plug-in; no fine-tuning) | No | Not mentioned | "Since our method is completely free plug-in and does not require fine-tuning" (Section 5 Experiments) |
| LongEval Lines Task | Not specified (plug-in; no fine-tuning) | No | Not mentioned | "Since our method is completely free plug-in and does not require fine-tuning" (Section 5 Experiments) |
| LongBench Single-Document QA | Yes (same model across LongBench tasks) | No | Not mentioned | "Accuracy on LongBench across multiple tasks using LLaMA2-7B-Chat." (Appendix C.4 Evaluation on LongBench); "Since our method is completely free plug-in and does not require fine-tuning" (Section 5 Experiments) |
| LongBench Multi-Document QA | Yes (same model across LongBench tasks) | No | Not mentioned | "Accuracy on LongBench across multiple tasks using LLaMA2-7B-Chat." (Appendix C.4 Evaluation on LongBench); "Since our method is completely free plug-in and does not require fine-tuning" (Section 5 Experiments) |
| LongBench Few-shot Learning | Yes (same model across LongBench tasks) | No | Not mentioned | "Accuracy on LongBench across multiple tasks using LLaMA2-7B-Chat." (Appendix C.4 Evaluation on LongBench); "Since our method is completely free plug-in and does not require fine-tuning" (Section 5 Experiments) |
| LongBench Synthesis Tasks | Yes (same model across LongBench tasks) | No | Not mentioned | "Accuracy on LongBench across multiple tasks using LLaMA2-7B-Chat." (Appendix C.4 Evaluation on LongBench); "Since our method is completely free plug-in and does not require fine-tuning" (Section 5 Experiments) |
| LongBench Code Completion | Yes (same model across LongBench tasks) | No | Not mentioned | "Accuracy on LongBench across multiple tasks using LLaMA2-7B-Chat." (Appendix C.4 Evaluation on LongBench); "Since our method is completely free plug-in and does not require fine-tuning" (Section 5 Experiments) |

## 6. Input and Representation Constraints

- Input length is variable: "Input: s[0:T-1] (input tokens with length T)" and the passkey dataset uses "samples of varying lengths." (Algorithm 1; Section 5.1 Evaluation on Passkey Retrieval Tasks)
- Fixed number of tokens: Not fixed; "The sample length initiates at 1024 and increments by 1024." (Section 5.1 Evaluation on Passkey Retrieval Tasks)
- Chunk size constraints: "In general, we set F = 100, M_max = 200 and L = 512." (Appendix B.2 Params Setting)
- Extrapolation position parameters: "we generally set the extrapolated position N=512 and set the extrapolated width E=50." (Appendix B.2 Params Setting)
- Fixed patch size: Not specified.
- Fixed dimensionality (e.g., strictly 2D): Not specified.
- Padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: "the position of maximum training length of the model. In this case, it is 4k for LLaMA2-7B-Chat model." (Figure 2 caption); "MPT-7B's maximum training length at 2k." (Figure 6 caption)
- Sequence length fixed or variable: Variable; "samples of varying lengths" and "input tokens with length T." (Section 5.1; Algorithm 1)
- Attention type: Chunked causal (triangular) attention; "We design a chunk-based triangular attention matrix." (Section 4.2 Chunk-based Triangular Attention Matrix)
- Mechanisms to manage computational cost: "To achieve approximate linear memory consumption and computational speed, we further split the triangular attention matrix into several chunks and concatenate these chunks." (Section 4.2 Chunk-based Triangular Attention Matrix)

## 8. Positional Encoding (Critical Section)

- Mechanism: Relative PE with weave variants; "We mainly consider relative PE methods" and "Stair PE can be applied to existing relative PEs such as RoPE and ALiBi." (Model Extrapolation: NoPE vs. Weave PE; Section 4.1 Stair PE)
- Where applied: In attention dot product / relative position handling; "self-attention dot product as a function f_PE" and "Notice that regular PE (such as RoPE or ALiBi) is applied to all chunks except for the last chunk, for which Stair PE is applied." (Model Extrapolation: NoPE vs. Weave PE; Section 4.3 Implementation)
- Fixed vs modified across experiments: Modified/compared; "We also provide an ablation experiment to compare these weave PE methods." (Section 4.1 Stair PE)

## 9. Positional Encoding as a Variable

- Core research variable: Yes; "PE is considered as a key factor influencing the extrapolating ability of LLMs." (Introduction)
- Multiple PEs compared: Yes; "we choose methods of this type for comparison, including: model self (Origin), ReRoPE [35], Leaky-ReRoPE [35], Dynamic-NTK [24], LM-Infinite [14], and Streaming-LLM [45]." (Section 5 Experiments)
- PE claimed as not critical or secondary: Not claimed.

## 10. Evidence of Constraint Masking

- Model sizes: "LLaMA [38] [39] (including LLaMA-3B (Open-LLaMA-3B), LLaMA2-7B-Chat, and Vicuna-13B-V1.3), MPT [37] (including MPT-7B), and PyThia [5] (including PyThia-6.9B and PyThia-12B)." (Section 5 Experiments)
- Dataset sizes / counts: "100 samples are randomly generated for each length." (Section 5.1 Evaluation on Passkey Retrieval Tasks); "A test set is created by randomly selecting 8 samples from each interval." (Section 5.3 Evaluation on Summary of Tasks)
- Attribution of gains: "LLMs equipped with weave PE can achieve improved extrapolation performance without additional cost" and Mesa-Extrapolation "utilizes a chunk-based triangular attention matrix and applies Stair PE to manage the final chunk." (Abstract)
- Scaling model size or data as primary driver: Not claimed.

## 11. Architectural Workarounds

- Chunk-based triangular attention matrix for efficiency: "We design a chunk-based triangular attention matrix... To achieve approximate linear memory consumption and computational speed, we further split the triangular attention matrix into several chunks and concatenate these chunks." (Section 4.2 Chunk-based Triangular Attention Matrix)
- DynamicSplit chunking of input: "We segment the input sequence into several sub-sequences according to DynamicSplit function" and it "divides a sequence into sub-sequences of equal length, with the exception of the first and last sequences." (Section 4.2 Chunk-based Triangular Attention Matrix)
- Stair PE on the last chunk: "regular PE (such as RoPE or ALiBi) is applied to all chunks except for the last chunk, for which Stair PE is applied." (Section 4.3 Implementation)
- K/V cache concatenation for last chunk: "For the last chunk, all previous chunks are concatenated, and Stair PE is used to rearrange relative positional encoding." (Section 4.3 Implementation)

## 12. Explicit Limitations and Non-Claims

- Limitations: "Due to limitations of resources, we have not yet validated our method at longer lengths." (Limitations)
- Future work: "Therefore exploring fine-tuning based on Mesa-Extrapolation can be an interesting next step." (Limitations)
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: text-only evaluations across multiple text subdomains (QA, summarization, code).
> - Task structure: fixed benchmark tasks (passkey retrieval, LongEval lines task, LongBench categories).
> - Representation rigidity: token sequences with max training length M and fixed chunk/PE parameters (F=100, L=512, N=512, E=50).
> - Model sharing vs specialization: plug-in approach with no fine-tuning; shared pretrained LLMs where stated.
> - Role of positional encoding: central research variable (weave PE / Stair PE).

### 14. Final Classification

Multi-task, multi-domain (constrained).
The paper evaluates multiple task categories, including "Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion," plus passkey retrieval ("We assess the accuracy of Mesa-Extrapolation using the generated passkey dataset.") and summarization ("We conduct a summary task using the GovReport dataset and employ ROUGE [31] (ROUGE-1/2/L) as evaluation metrics."). These evaluations are constrained to specific benchmarks and long-text settings, and no cross-domain generalization claim is made.
