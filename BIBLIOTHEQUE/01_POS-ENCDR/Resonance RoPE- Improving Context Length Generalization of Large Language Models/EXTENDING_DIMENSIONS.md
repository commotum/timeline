## 1. Basic Metadata

- Title: "Resonance RoPE: Improving Context Length Generalization of Large Language Models" (Title line)
- Authors: "Suyuchen Wang<sup>1,2</sup>, Ivan Kobyzev<sup>3</sup>, Peng Lu<sup>1</sup>, Mehdi Rezagholizadeh<sup>3</sup> and Bang Liu<sup>1,2†</sup>" (Title line)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper introduces "RESONANCE ROPE, a novel technique designed to further narrow the generalization gap on position embeddings in TSTL scenarios" (Section 1 Introduction).

---

## 3. Tasks Evaluated

Task 1: POSGEN - Recursive subtask (synthetic next-token prediction)
Task type: Generation
Dataset(s) used: POSGEN synthetic sequences; modular addition task
Domain: Synthetic token sequences
Quotes: "we present PosGEN, a new synthetic benchmark specifically designed for fine-grained behavior analysis in TSTL scenarios" (Abstract) "Our PosGeN framework comprises three subtasks, with each extracting the general token dependency pattern of a different type of reasoning task." (Section 5) "**Recursive.** This task simulates the token dependency pattern of generating a Fibonaccistyle sequence, where new tokens depend on j+k neighboring tokens only:" (Section 5) "We test on a modular addition task, which was proved to be learnable by a one-layer Transformer (Nanda et al., 2023)." (Section 6.1.1)

Task 2: POSGEN - Chain-of-Thought (CoT) subtask
Task type: Generation; Reasoning / relational
Dataset(s) used: POSGEN synthetic sequences; modular addition task
Domain: Synthetic token sequences
Quotes: "we present PosGEN, a new synthetic benchmark specifically designed for fine-grained behavior analysis in TSTL scenarios" (Abstract) "Our PosGeN framework comprises three subtasks, with each extracting the general token dependency pattern of a different type of reasoning task." (Section 5) "Chain-of-Thought (CoT). This task simulates the token dependency pattern of CoT reasoning (Wei et al., 2022), where new tokens depend on k neighboring tokens (simulating the previous reasoning step) and j tokens in the front (simulating the original question):" (Section 5) "We test on a modular addition task, which was proved to be learnable by a one-layer Transformer (Nanda et al., 2023)." (Section 6.1.1)

Task 3: POSGEN - Semi-recursive subtask
Task type: Generation; Reasoning / relational
Dataset(s) used: POSGEN synthetic sequences; modular addition task
Domain: Synthetic token sequences
Quotes: "we present PosGEN, a new synthetic benchmark specifically designed for fine-grained behavior analysis in TSTL scenarios" (Abstract) "Our PosGeN framework comprises three subtasks, with each extracting the general token dependency pattern of a different type of reasoning task." (Section 5) "**Semi-recursive.** This task simulates the token dependency pattern of the last-letter concatenation task (Zhou et al., 2023), where new tokens depend on both k neighboring tokens (simulating the current progress) and j tokens with varied distances according to a specific rule (simulating the word sequence):" (Section 5) "We test on a modular addition task, which was proved to be learnable by a one-layer Transformer (Nanda et al., 2023)." (Section 6.1.1)

Task 4: Language modeling perplexity on long-text corpora
Task type: Generation
Dataset(s) used: GovReport; Proofpile
Domain: Long-text sequences
Quotes: "We evaluate the model's language modeling performance on GovReport (Huang et al., 2021) and Proofpile (Azerbayev, 2022)." (Section 6.2.2) "We test the model's performance on two TSTL scenarios: language modeling evaluation on long-text sequences and long-text downstream application performance." (Section 6.2.1)

Task 5: L-Eval close ended tasks (Coursera, GSM, QuALITY, TOEFL, CodeU, SFiction)
Task type: Other (close ended long-text benchmark)
Dataset(s) used: L-Eval close ended task suite (Coursera, GSM, QuALITY, TOEFL, CodeU, SFiction)
Domain: Long-text LLM benchmark covering multiple domains
Quotes: "we test the real-world task performance of LLaMA2-Chat 7B and 13B's performance with different RoPE scaling strategies on L-Eval (An et al., 2023)'s close ended task suite, a long-text LLM benchmark covering a wide range of domains such as school lectures, long conversations and novels." (Section 6.2.3) "| Setting                                        | Ctx Len. | Coursera     | GSM          | QuALITY | TOEFL        | CodeU       | SFiction     | Avg.  |" (Table 2)

---

## 4. Domain and Modality Scope

- Single domain? No. "a long-text LLM benchmark covering a wide range of domains such as school lectures, long conversations and novels." (Section 6.2.3)
- Multiple domains within the same modality? Yes. "a long-text LLM benchmark covering a wide range of domains such as school lectures, long conversations and novels." (Section 6.2.3)
- Multiple modalities? Not stated; evaluations are described as "long-text sequences" (Section 6.2.1) and a "next token prediction task" (Section 5).
- Does the paper claim domain generalization or cross-domain transfer? Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| POSGEN - Recursive | No (trained per subtask) | Not specified (trained per subtask) | Not specified | "we train a two-layer Transformer on each of the subtasks" (Appendix C.1) |
| POSGEN - Chain-of-Thought (CoT) | No (trained per subtask) | Not specified (trained per subtask) | Not specified | "we train a two-layer Transformer on each of the subtasks" (Appendix C.1) |
| POSGEN - Semi-recursive | No (trained per subtask) | Not specified (trained per subtask) | Not specified | "we train a two-layer Transformer on each of the subtasks" (Appendix C.1) |
| Language modeling perplexity (GovReport, Proofpile) | Yes (same model setting evaluated across tasks) | Yes for YaRN/Resonance settings; "no FT" baselines also evaluated | Not specified | "For the configurations that require fine-tuning, we fine-tune the LLM with the scaled position embedding on the training set of PG19 (Rae et al., 2020) with the fine-tuning setting and hyperparameters adopted directly from YaRN (Peng et al., 2024), with the only difference being that we control the total training token count to be approximately 100M." (Section 6.2.1) "We test the model's performance on two TSTL scenarios: language modeling evaluation on long-text sequences and long-text downstream application performance." (Section 6.2.1) "The settings with "no FT" are not fine-tuned after modifying its position embedding." (Table 2) |
| L-Eval close ended tasks (Coursera, GSM, QuALITY, TOEFL, CodeU, SFiction) | Yes (same model setting evaluated across tasks) | Yes for YaRN/Resonance settings; "no FT" baselines also evaluated | Not specified | "For the configurations that require fine-tuning, we fine-tune the LLM with the scaled position embedding on the training set of PG19 (Rae et al., 2020) with the fine-tuning setting and hyperparameters adopted directly from YaRN (Peng et al., 2024), with the only difference being that we control the total training token count to be approximately 100M." (Section 6.2.1) "We test the model's performance on two TSTL scenarios: language modeling evaluation on long-text sequences and long-text downstream application performance." (Section 6.2.1) "The settings with "no FT" are not fine-tuned after modifying its position embedding." (Table 2) |

---

## 6. Input and Representation Constraints

- Sequence length and per-head dimensionality: "Suppose the input to a single attention head is  $x_1, x_2, \ldots, x_l \in \mathbb{R}^d$ , where l is the sequence length and d is the dimension of an attention head." (Section 3.1)
- Fixed sequence lengths for POSGEN training/testing: "The models are trained on sequences of length L=64, and evaluating on lengths of L' = 256 for OOD Accuracy." (Section 6.1.1)
- Fixed task parameters and vocabulary for POSGEN: "We configured j=1, k=3" and "with vocabulary  $\mathbb{V} = \{0, \dots, 16\}$ ." (Section 6.1.1)
- Fixed sequence lengths for LLM fine-tuning strategies: "we fine-tune the model on sequences with length 32,768 for 50 steps and sequences with length 16,384 for 100 steps" and "we fine-tune the model on sequences with length 4,096 for 400 steps." (Appendix C.2)
- Padding/resizing requirements: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length reported in experiments: "For YaRN and RESONANCE YARN, We use a scaling factor of 8 and 4 for LLaMA2 7B and 13B to extend their context window from 4K to 32K and 16K, respectively." (Section 6.2.1)
- Synthetic task context lengths: "The models are trained on sequences of length L=64, and evaluating on lengths of L' = 256 for OOD Accuracy." (Section 6.1.1)
- Fixed vs variable length: "We randomly select 50 samples from each dataset and report the final perplexity in text fragments of gradually increased length." (Section 6.2.2)
- Attention type: Not explicitly specified beyond standard Transformer self-attention: "In Transformers (Vaswani et al., 2017), the self-attention scores are softmax-normalized scaled attention logits  $q^{\top}k$" (Section 3.1).
- Mechanisms for computational cost: No explicit windowing/pooling/sparse attention described; the method claims no extra runtime cost: "without additional online computational costs." (Abstract)

---

## 8. Positional Encoding (Critical Section)

- Mechanism used: RoPE is the base positional encoding: "RoPE injects the position information of each token into the q and k vectors" (Section 3.1) and the paper "we introduce RESONANCE ROPE, a novel technique designed to further narrow the generalization gap on position embeddings in TSTL scenarios." (Section 1 Introduction).
- Where applied: "RoPE injects the position information of each token into the q and k vectors" (Section 3.1).
- Fixed vs modified across experiments: The paper compares multiple RoPE-based variants: "We compare the same RoPE-based PE with or without our RESONANCE scaling." (Table 1) and "More specifically, we replace the original position embeddings of LLaMA2 7B and 13B (Touvron et al., 2023b) with a series of scaled position embeddings, including the NTK-Aware scaling (bloc97, 2023; Xiong et al., 2023; Liu et al., 2024), Dynamic NTK-Aware Scaling" (Section 6.2.1) and "In this section, we apply our proposed RESO-NANCE ROPE to the current state-of-the-art RoPE scaling method, YaRN (Peng et al., 2024)." (Section 6.2.1)
- Input-only vs every layer vs attention bias: Not specified.

---

## 9. Positional Encoding as a Variable

- Core research variable? Yes. "We propose RESONANCE ROPE, an innovative modification to RoPE" (Section 1 Introduction).
- Multiple positional encodings compared? Yes. "We compare the same RoPE-based PE with or without our RESONANCE scaling." (Table 1) and "More specifically, we replace the original position embeddings of LLaMA2 7B and 13B (Touvron et al., 2023b) with a series of scaled position embeddings, including the NTK-Aware scaling (bloc97, 2023; Xiong et al., 2023; Liu et al., 2024), Dynamic NTK-Aware Scaling" (Section 6.2.1) and "In this section, we apply our proposed RESO-NANCE ROPE to the current state-of-the-art RoPE scaling method, YaRN (Peng et al., 2024)." (Section 6.2.1)
- Claim that PE choice is not critical or secondary? Not stated.

---

## 10. Evidence of Constraint Masking

- Model sizes: "we replace the original position embeddings of LLaMA2 7B and 13B (Touvron et al., 2023b)" (Section 6.2.1) and "Our experiments involved training a two-layer Transformer." (Section 6.1.1)
- Dataset sizes: "We generated 10,000 training sequences, and 1,000 each for validation and testing" (Section 6.1.1) and "we control the total training token count to be approximately 100M." (Section 6.2.1)
- Attribution of gains: The paper attributes improvements to PE interpolation reduction: "This improvement indicates a superior adaptation to OOD position embeddings through minimized Positional Encoding (PE) interpolation." (Section 6.1.2)
- Scaling model size/data as primary driver? Not claimed; improvements are tied to positional encoding changes and "without additional online computational costs." (Abstract)

---

## 11. Architectural Workarounds

- RoPE feature scaling strategies: 'It introduces the "NTK-by-parts" scaling for RoPE, which applies different scaling strategies to each RoPE feature according to its temporal wavelength.' (Section 3.3)
- Attention score scaling in YaRN: "YaRN also comprises a scaling strategy on the attention scores, which reduces the change in the entropy of the attention score on longer sequences." (Section 3.3)
- Resonance RoPE modification: "we round their wavelengths to their nearest integer to eliminate new rotary angles on each feature." (Section 4)
- Replacing PE variants for LLM evaluation: "More specifically, we replace the original position embeddings of LLaMA2 7B and 13B (Touvron et al., 2023b) with a series of scaled position embeddings, including the NTK-Aware scaling (bloc97, 2023; Xiong et al., 2023; Liu et al., 2024), Dynamic NTK-Aware Scaling" (Section 6.2.1) and "In this section, we apply our proposed RESO-NANCE ROPE to the current state-of-the-art RoPE scaling method, YaRN (Peng et al., 2024)." (Section 6.2.1)

---

## 12. Explicit Limitations and Non-Claims

- Post-critical extrapolation not solved: "this method does not solve the extrapolation issue on RoPE's post-critical dimensions" (Limitations)
- Focus limited to TSTL performance, not efficiency: "we focus only on improving Transformers' performance in TSTL scenarios." (Limitations)
- Efficiency is future work: "An interesting future direction would be to apply RESONANCE ROPE to efficient Transformers for both performance and efficiency enhancements." (Limitations)
- Benchmarking limitations: "benchmarking LLMs is still an open question, as there is currently no benchmark to thoroughly test the performance of LLMs, especially on long-sequence tasks." (Limitations)

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Synthetic token sequences plus long-text natural language across multiple domains (L-Eval).
> - Task structure: Next-token prediction POSGEN subtasks, language modeling perplexity, and close ended long-text evaluations.
> - Representation rigidity: Fixed sequence lengths per experiment (L=64, L'=256; 32K/16K/4,096) and fixed j,k and vocabulary; RoPE applied to q/k.
> - Model sharing vs specialization: Separate two-layer Transformer per POSGEN subtask; shared LLaMA2 weights across multiple evaluations after PE scaling.
> - Role of positional encoding: Central variable with RoPE vs YaRN vs Resonance comparisons.

---

### 14. Final Classification

Classification: **Multi-task, multi-domain (constrained)**.
The paper evaluates "three different TSTL tasks" spanning POSGEN synthetic tasks and LLM-scale evaluations "on both language modeling perplexity and real-world long context applications" (Section 6), and it uses L-Eval, described as "a long-text LLM benchmark covering a wide range of domains such as school lectures, long conversations and novels." (Section 6.2.3) This indicates multiple domains within the text modality while remaining constrained to specific long-text and synthetic tasks.
