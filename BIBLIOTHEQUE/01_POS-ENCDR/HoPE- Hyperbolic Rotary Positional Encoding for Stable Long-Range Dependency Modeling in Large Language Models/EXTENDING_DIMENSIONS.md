## 1. Basic Metadata

- Title: "HoPE: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models" (Title)
- Authors: "Chang Dai<sup>1</sup> Hongyu Shan<sup>2</sup> Mingyang Song<sup>3</sup> Di liang<sup>4</sup>" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

"Drawing inspiration from Lorentz transformations in hyperbolic geometry, we propose Hyperbolic Rotary Positional Encoding (HoPE), which leverages hyperbolic functions to implement Lorentz rotations on token representations." (Abstract)

---

## 3. Tasks Evaluated

- Task name: Perplexity evaluation / language modeling (length extrapolation)
  - Task type: Other (specify): Language modeling (next-token prediction)
  - Dataset(s) used: Pile (pre-training), PG19 (test), arXiv (test)
  - Domain: Text (natural language)
  - Evidence: "We test the length extrapolation capability of Transformer-based language models with various positional encoding methods. Following the methodology of (Chi et al., 2022b), we use the Pile dataset (Gao et al., 2020)as the pre-training corpus and evaluate the log perplexity of pre-trained language models in the test sets of PG19 (Rae et al., 2019) and arXiv." (Section 4.2 Perplexity Experiment (PPL)) "The next token prediction objective is adopted for language model training." (Section 8.3.1 Perplexity Experiment)

- Task name: Qasper (Question-Answering)
  - Task type: Other (specify): Question-Answering
  - Dataset(s) used: Qasper (SCROLLS)
  - Domain: Text (natural language)
  - Evidence: "It is a long context benchmark that consists of seven distinct datasets covering different tasks, e.g, Question-Answering (Qasper(Dasigi et al., 2021), NarrativeQA(Kočiský et al., 2017), and QuALITY(Pang et al., 2022))" (Section 8.3.2 Fine-Tuning Experiment)

- Task name: NarrativeQA (Question-Answering)
  - Task type: Other (specify): Question-Answering
  - Dataset(s) used: NarrativeQA (SCROLLS)
  - Domain: Text (natural language)
  - Evidence: "It is a long context benchmark that consists of seven distinct datasets covering different tasks, e.g, Question-Answering (Qasper(Dasigi et al., 2021), NarrativeQA(Kočiský et al., 2017), and QuALITY(Pang et al., 2022))" (Section 8.3.2 Fine-Tuning Experiment)

- Task name: QuALITY (Question-Answering)
  - Task type: Other (specify): Question-Answering
  - Dataset(s) used: QuALITY (SCROLLS)
  - Domain: Text (natural language)
  - Evidence: "It is a long context benchmark that consists of seven distinct datasets covering different tasks, e.g, Question-Answering (Qasper(Dasigi et al., 2021), NarrativeQA(Kočiský et al., 2017), and QuALITY(Pang et al., 2022))" (Section 8.3.2 Fine-Tuning Experiment)

- Task name: ContractNLI (Natural Language Inference)
  - Task type: Other (specify): Natural Language Inference
  - Dataset(s) used: ContractNLI (SCROLLS)
  - Domain: Text (natural language)
  - Evidence: "It is a long context benchmark that consists of seven distinct datasets covering different tasks, e.g, ... Natural Language Inference (ContractNLI(Koreeda and Manning, 2021))" (Section 8.3.2 Fine-Tuning Experiment)

- Task name: QMSum (Summarization)
  - Task type: Other (specify): Summarization
  - Dataset(s) used: QMSum (SCROLLS)
  - Domain: Text (natural language)
  - Evidence: "It is a long context benchmark that consists of seven distinct datasets covering different tasks, e.g, ... Summarization (QMSum(Zhong et al., 2021), SummScreenFD(Chen et al., 2022), and GovReport(Huang et al., 2021))." (Section 8.3.2 Fine-Tuning Experiment)

- Task name: SummScreenFD (Summarization)
  - Task type: Other (specify): Summarization
  - Dataset(s) used: SummScreenFD (SCROLLS)
  - Domain: Text (natural language)
  - Evidence: "It is a long context benchmark that consists of seven distinct datasets covering different tasks, e.g, ... Summarization (QMSum(Zhong et al., 2021), SummScreenFD(Chen et al., 2022), and GovReport(Huang et al., 2021))." (Section 8.3.2 Fine-Tuning Experiment)

- Task name: GovReport (Summarization)
  - Task type: Other (specify): Summarization
  - Dataset(s) used: GovReport (SCROLLS)
  - Domain: Text (natural language)
  - Evidence: "It is a long context benchmark that consists of seven distinct datasets covering different tasks, e.g, ... Summarization (QMSum(Zhong et al., 2021), SummScreenFD(Chen et al., 2022), and GovReport(Huang et al., 2021))." (Section 8.3.2 Fine-Tuning Experiment)

---

## 4. Domain and Modality Scope

- Modality/domain scope: The paper evaluates on text-only tasks; multiple datasets within the text modality are used. Evidence: "First, while it excels in text-only tasks, its performance in multimodal scenarios (where text, audio, and visual inputs must be jointly modelled) remains unverified." (Section 7 Limitations) "We use the Pile dataset (Gao et al., 2020)as the pre-training corpus and evaluate the log perplexity of pre-trained language models in the test sets of PG19 (Rae et al., 2019) and arXiv." (Section 4.2 Perplexity Experiment (PPL)) "It is a long context benchmark that consists of seven distinct datasets covering different tasks" (Section 8.3.2 Fine-Tuning Experiment)
- Domain generalization or cross-domain transfer claimed?: Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
|---|---|---|---|---|
| Perplexity evaluation / language modeling (PG19, arXiv) | Yes (same pre-trained model evaluated) | No (zero-shot) | Not specified | "we use the Pile dataset (Gao et al., 2020)as the pre-training corpus and evaluate the log perplexity of pre-trained language models in the test sets of PG19 (Rae et al., 2019) and arXiv." (Section 4.2 Perplexity Experiment (PPL)) "we evaluate zero-shot perplexity on sequence lengths [1024, 2048, 3072, 4096, 5120, 6144]." (Section 4.2 Perplexity Experiment (PPL)) |
| Qasper (QA) | Not shared (fine-tuned per task) | Yes | Not specified | "We fine-tune models using the next token prediction objective on each task with a sequence length of 8192." (Section 8.3.2 Fine-Tuning Experiment) |
| NarrativeQA (QA) | Not shared (fine-tuned per task) | Yes | Not specified | "We fine-tune models using the next token prediction objective on each task with a sequence length of 8192." (Section 8.3.2 Fine-Tuning Experiment) |
| QuALITY (QA) | Not shared (fine-tuned per task) | Yes | Not specified | "We fine-tune models using the next token prediction objective on each task with a sequence length of 8192." (Section 8.3.2 Fine-Tuning Experiment) |
| ContractNLI (NLI) | Not shared (fine-tuned per task) | Yes | Not specified | "We fine-tune models using the next token prediction objective on each task with a sequence length of 8192." (Section 8.3.2 Fine-Tuning Experiment) |
| QMSum (Summarization) | Not shared (fine-tuned per task) | Yes | Not specified | "We fine-tune models using the next token prediction objective on each task with a sequence length of 8192." (Section 8.3.2 Fine-Tuning Experiment) |
| SummScreenFD (Summarization) | Not shared (fine-tuned per task) | Yes | Not specified | "We fine-tune models using the next token prediction objective on each task with a sequence length of 8192." (Section 8.3.2 Fine-Tuning Experiment) |
| GovReport (Summarization) | Not shared (fine-tuned per task) | Yes | Not specified | "We fine-tune models using the next token prediction objective on each task with a sequence length of 8192." (Section 8.3.2 Fine-Tuning Experiment) |

---

## 6. Input and Representation Constraints

- Fixed/variable input length constraints: "The pre-training sequence length is set to 1024" (Section 4.2 Perplexity Experiment (PPL)); "we evaluate zero-shot perplexity on sequence lengths [1024, 2048, 3072, 4096, 5120, 6144]." (Section 4.2 Perplexity Experiment (PPL)); "We fine-tune pre-trained models using a sequence length of 8192" (Section 4.3 Fine-Tuning Experiment);
- Token-based input assumption: "Let  $\mathbb{S}_N = \{w_i\}_{i=1}^N$  be a sequence of N input tokens" (Section 2.1 Relative position encoding)
- Fixed patch size, fixed resolution, fixed dimensionality (e.g., 2D grids): Not specified.
- Padding or resizing requirements: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length used: "We fine-tune pre-trained models using a sequence length of 8192" (Section 4.3 Fine-Tuning Experiment)
- Fixed vs. variable sequence length: "The pre-training sequence length is set to 1024" (Section 4.2 Perplexity Experiment (PPL)); "we evaluate zero-shot perplexity on sequence lengths [1024, 2048, 3072, 4096, 5120, 6144]." (Section 4.2 Perplexity Experiment (PPL)); "We fine-tune pre-trained models using a sequence length of 8192" (Section 4.3 Fine-Tuning Experiment)
- Attention type (global/windowed/hierarchical/sparse): Not specified. (Only "We choose the standard decoder-only Transformer(Touvron et al., 2023) as the base model" is stated.) (Section 4.2 Perplexity Experiment (PPL))
- Computational cost mechanisms (windowing/pooling/pruning): Not specified.

---

## 8. Positional Encoding (Critical Section)

- Mechanism used: "we propose Hyperbolic Rotary Positional Encoding (HoPE), which leverages hyperbolic functions to implement Lorentz rotations on token representations." (Abstract)
- Where applied: "Like RoPE, we apply rotations of  $m\theta$  and  $-m\theta$  to the query (q) and key (k) vectors at position m, respectively." (Section 3.1 Hyperbolic Rotary Position Encoding)
- Compared/ablated: "we choose the standard decoder-only Transformer(Touvron et al., 2023) as the base model and compare our HoPE method against other positional encoding methods: RoPE and Alibi" (Section 4.2 Perplexity Experiment (PPL)); "To further analyze HoPE's effectiveness, we conduct ablation studies that examine the impact of individual components." (Section 4.4 Ablation Studies)
- Fixed vs modified per task: Positional encoding varies by method across experiments (HoPE vs RoPE vs Alibi vs others), but per-task modification is not explicitly described. Evidence: "we ... compare our HoPE method against other positional encoding methods: RoPE and Alibi" (Section 4.2 Perplexity Experiment (PPL)); "Table 3: Performance comparison on SCROLLS benchmark." (Section 4.3 Fine-Tuning Experiment)

---

## 9. Positional Encoding as a Variable

- Core research variable vs fixed assumption: It is a core research variable. Evidence: "we propose Hyperbolic Rotary Positional Encoding (HoPE)" (Abstract) and "we ... compare our HoPE method against other positional encoding methods: RoPE and Alibi" (Section 4.2 Perplexity Experiment (PPL))
- Multiple positional encodings compared: Yes. Evidence: "compare our HoPE method against other positional encoding methods: RoPE and Alibi" (Section 4.2 Perplexity Experiment (PPL)) and "Table 3: Performance comparison on SCROLLS benchmark" (Section 4.3 Fine-Tuning Experiment)
- Claims that PE choice is not critical/secondary: Not claimed.

---

## 10. Evidence of Constraint Masking

- Model size(s): "The Transformer-based language model configuration includes 12 layers, a hidden dimension of 768, and 12 attention heads, resulting in approximately 155M parameters." (Section 4.2 Perplexity Experiment (PPL))
- Dataset size(s): Not specified for Pile, PG19, arXiv, or SCROLLS.
- Attribution of gains (scaling vs architecture/training): Gains are attributed to positional encoding design rather than scaling. Evidence: "Extensive experimental results, including perplexity evaluations under several extended sequence benchmarks, show that HoPE consistently exceeds existing positional encoding methods." (Abstract)

---

## 11. Architectural Workarounds

- Hyperbolic rotation with decay penalty to enforce monotonic attention: "we introduce a penalty coefficient  $e^{\pm m\theta'}$ , where  $\theta'$  is a learnable or predefined parameter, to modulate the positional impact on the dot product." (Section 3.1 Hyperbolic Rotary Position Encoding)
- Other workarounds (windowed attention, hierarchical stages, token pooling, task-specific heads, fixed grid assumptions): Not specified.

---

## 12. Explicit Limitations and Non-Claims

- Limitations: "First, while it excels in text-only tasks, its performance in multimodal scenarios (where text, audio, and visual inputs must be jointly modelled) remains unverified. Second, the method's effectiveness hinges on careful tuning of the damping coefficient  $\theta'$ . Suboptimal choices can degrade performance, especially for tasks with varying positional sensitivity requirements." (Section 7 Limitations)
- Explicit non-claims about open-world learning or unrestrained multi-task learning: Not specified.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Text-only evaluation across multiple text datasets; multimodal performance is explicitly unverified.
> - Task structure: Multiple NLP tasks (language modeling/perplexity, QA, NLI, summarization) evaluated via SCROLLS and long-context benchmarks.
> - Representation rigidity: Fixed sequence lengths in training/fine-tuning (1024, 8192) with specified evaluation lengths; token-sequence input assumed.
> - Model sharing vs specialization: Single pre-trained model evaluated for perplexity; fine-tuning performed "on each task" for SCROLLS tasks.
> - Role of positional encoding: Central experimental variable with multiple encodings compared and ablations on components.

---

### 14. Final Classification

**Multi-task, single-domain.** The evaluation covers multiple tasks (language modeling, QA, NLI, summarization) but all are text-only and within a single modality, as stated by "text-only tasks" and the SCROLLS task list. The paper does not claim cross-domain or multimodal transfer; it explicitly notes multimodal performance is unverified.
