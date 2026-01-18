## 1. Basic Metadata

Title: "Positional Encoding via Token-Aware Phase Attention" (Title block)
Authors: "Yu (Sid) Wang<sup>1</sup>, Sheng Shen\*, Rémi Munos<sup>1</sup>, Hongyuan Zhan<sup>1</sup>, Yuandong Tian<sup>1</sup>" (Title block)
Year: Year not specified.
Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

"This paper introduces Token-Aware Phase Attention (TAPA), a new positional encoding method that incorporates a learnable phase function into the attention mechanism." (Opening paragraph before Section 1 Introduction)

## 3. Tasks Evaluated

Task 1: Long-context language modeling (perplexity evaluation).
- Task type: Generation.
- Dataset(s) used: PG19 (train split for fine-tuning; test split for evaluation); Pile for pretraining.
- Domain: Text (long sequence language modeling).
- Quotes: "RoPE's distance bias is harmful for long-context language modeling, as it hurts model's ability in feeling long-range dependencies and leveraging distant information." (Section 3 A New Positional Encoding: Token-Aware Phase Attention (TAPA)); "We evaluate all fine-tuned models on the test split of PG19 (Rae et al., 2019) which consist of mostly long sequence samples." (Section 4.3 Long-Context Evaluation); "The pretraining uses Pile (Gao et al., 2020), and each training document is chunked into 8k length segments." (Section 4.1 Pretraining); "To extend to long context, we further fine-tune pretrained models with different Positional Encoding methods on the training split of PG19 (Rae et al., 2019), where each document is chunked into segments of length 32k." (Section 4.2 Long-Context Fine-tuning); "Test perplexities on PG19 test set of LLaMA3 7B transformers first pretrained on 8k context length and further fine-tuned on 32k, and evaluated on  $1k\sim64k$ ." (Table 1 caption)

## 4. Domain and Modality Scope

- Evaluation domain: Single domain (text language modeling) supported by "RoPE's distance bias is harmful for long-context language modeling, as it hurts model's ability in feeling long-range dependencies and leveraging distant information." (Section 3 A New Positional Encoding: Token-Aware Phase Attention (TAPA)) and "We evaluate all fine-tuned models on the test split of PG19 (Rae et al., 2019) which consist of mostly long sequence samples." (Section 4.3 Long-Context Evaluation).
- Multiple domains within the same modality: Not indicated beyond PG19; evaluation is on PG19: "We evaluate all fine-tuned models on the test split of PG19 (Rae et al., 2019) which consist of mostly long sequence samples." (Section 4.3 Long-Context Evaluation).
- Multiple modalities: Not indicated; the paper frames the work as "long-context language modeling" (Section 3 A New Positional Encoding: Token-Aware Phase Attention (TAPA)).
- Domain generalization or cross-domain transfer: Not claimed. The paper instead emphasizes length extrapolation: "extrapolates to unseen lengths" (Opening paragraph before Section 1 Introduction).

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Long-context language modeling (PG19 perplexity) | N/A (single task) | Yes | Not specified | "We pretrain Transformers with LLaMA3 7B architecture on 8k context length, fine-tune on 32k, and evaluate up to 64k." (Section 4 Experiments); "To extend to long context, we further fine-tune pretrained models with different Positional Encoding methods on the training split of PG19 (Rae et al., 2019), where each document is chunked into segments of length 32k." (Section 4.2 Long-Context Fine-tuning) |

## 6. Input and Representation Constraints

- Pretraining length is fixed at 8k segments: "The pretraining uses Pile (Gao et al., 2020), and each training document is chunked into 8k length segments." (Section 4.1 Pretraining)
- Fine-tuning length is fixed at 32k segments: "To extend to long context, we further fine-tune pretrained models with different Positional Encoding methods on the training split of PG19 (Rae et al., 2019), where each document is chunked into segments of length 32k." (Section 4.2 Long-Context Fine-tuning)
- Evaluation uses variable context windows: "with context window size varying from 1k to 64k in the dyadic fashion." (Section 4.3 Long-Context Evaluation)
- Evaluation uses a sliding window stride: "For each context window, we closely follow the sliding window method from (Press et al., 2021) with stride = 256 to calculate the test loss." (Section 4.3 Long-Context Evaluation)
- Fixed input resolution, patch size, 2D dimensionality assumptions, padding, or resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: "evaluate up to 64k." (Section 4 Experiments)
- Sequence length fixed vs variable: "We pretrain Transformers with LLaMA3 7B architecture on 8k context length, fine-tune on 32k, and evaluate up to 64k." (Section 4 Experiments); "with context window size varying from 1k to 64k in the dyadic fashion." (Section 4.3 Long-Context Evaluation)
- Attention type: Not explicitly specified; the paper states "we preserve the causal masking mechanism" (Section 5.3 Non-RoPE Approaches to Positional Extrapolation).
- Mechanisms for computational cost: Not specified for attention; evaluation uses sliding windows for loss computation: "For each context window, we closely follow the sliding window method from (Press et al., 2021) with stride = 256 to calculate the test loss." (Section 4.3 Long-Context Evaluation).

## 8. Positional Encoding (Critical Section)

- Mechanism used: TAPA with a learnable phase in attention: "This paper introduces Token-Aware Phase Attention (TAPA), a new positional encoding method that incorporates a learnable phase function into the attention mechanism." (Opening paragraph before Section 1 Introduction) and "$$\operatorname{Attn}_{\phi,\mathcal{M},\alpha}(q,k) = q^{\top} \mathcal{M}k \cdot \cos\left(2\pi |m-n|^{\alpha}\phi(q,k)\right). \tag{9}$$" (Section 3 A New Positional Encoding: Token-Aware Phase Attention (TAPA)).
- Where applied: In attention computation; "inserts a learnable phase function into the attention mechanism" (Opening paragraph before Section 1 Introduction) and "replacing transformer's inner product attention with Equation (12)" (Section 4.1 Pretraining).
- Compared/ablated across experiments: Multiple positional encodings are evaluated and TAPA phase choices are compared: "We report the test perplexity for multiple Positional Encoding methods on context window sizes ranging from 1k to 64k on the checkpoints obtained from Subsection 4.2." (Section 4.4 Evaluation Results); "We compare two phase functions for TAPA: (i) *quadratic* (stationary) phase in (10) and (ii) *linear* (non-stationary) phase:" (Section 4.5 Ablations: TAPA's phase choice). For TAPA fine-tuning, the method is held fixed: "In TAPA fine-tuning, we keep all hyper-parameters, architectures, and attention computations the same from pretraining." (Section 4.2 Long-Context Fine-tuning)

## 9. Positional Encoding as a Variable

- Core research variable: "For fair comparison, we pretrain Transformers with LLaMA3 7B (Dubey et al., 2024) architecture from scratch with TAPA (3.1) and RoPE (Su et al., 2021) respectively." (Section 4.1 Pretraining)
- Multiple positional encodings compared: "We report the test perplexity for multiple Positional Encoding methods on context window sizes ranging from 1k to 64k on the checkpoints obtained from Subsection 4.2." (Section 4.4 Evaluation Results)
- Additional PE variants within TAPA: "We compare two phase functions for TAPA: (i) *quadratic* (stationary) phase in (10) and (ii) *linear* (non-stationary) phase:" (Section 4.5 Ablations: TAPA's phase choice)
- Claim that PE choice is not critical or secondary: Not claimed.

## 10. Evidence of Constraint Masking

- Model size: "We pretrain Transformers with LLaMA3 7B (Dubey et al., 2024) architecture from scratch" (Section 4.1 Pretraining).
- Dataset scale: "The pretraining uses  $512 \times H100$  GPUs with a global batch size of 256 for a total of 200k steps, which results in a total of 420B tokens." (Section 4.1 Pretraining)
- Dataset source: "The pretraining uses Pile (Gao et al., 2020)" (Section 4.1 Pretraining).
- Gains attributed to PE change and light fine-tuning: "We make no changes to transformer architecture other than removing RoPE and replacing transformer's inner product attention with Equation (12)" (Section 4.1 Pretraining); "Our experiments show that TAPA is able to adapt to 32k by only fine-tuning on less than 0.25% of pretraining tokens" (Section 4 Experiments); "TAPA eliminates undesired distance bias and preserves interactions with long-range context." (Section 3 A New Positional Encoding: Token-Aware Phase Attention (TAPA))

## 11. Architectural Workarounds

- No additional parameters or flops beyond a vanilla transformer: "One notable benefit of the form (12) is that no new parameters or flops are introduced in addition to those of a vanilla transformer." (Section 3 A New Positional Encoding: Token-Aware Phase Attention (TAPA))
- Architecture otherwise unchanged aside from attention replacement: "we make no changes to transformer architecture other than removing RoPE and replacing transformer's inner product attention with Equation (12)" (Section 4.1 Pretraining)
- Causal masking retained (no alternative attention structure stated): "we preserve the causal masking mechanism and focus on improving the positional encoding itself." (Section 5.3 Non-RoPE Approaches to Positional Extrapolation)

## 12. Explicit Limitations and Non-Claims

- Implementation limitation/future work: "While a flash-style implementation is feasible—e.g., via PyTorch FlexAttention with an extra  $QK^{\top}$  matmul or a custom Triton kernel, these engineering focus is beyond the scope of this work and we leave it for future study." (Section 4.1 Pretraining)
- Non-claim about other architectures: "Since these methods deviate from the standard transformer architecture, whereas our work assumes the traditional attention, we do not discuss them in depth." (Section 5.3 Non-RoPE Approaches to Positional Extrapolation)

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: single text domain, "test split of PG19 (Rae et al., 2019) which consist of mostly long sequence samples." (Section 4.3 Long-Context Evaluation)
> - Task structure: fixed language modeling focus, "long-context language modeling" (Section 3 A New Positional Encoding: Token-Aware Phase Attention (TAPA))
> - Representation rigidity: fixed-length chunking and bounded evaluation windows, "each training document is chunked into 8k length segments." (Section 4.1 Pretraining); "each document is chunked into segments of length 32k." (Section 4.2 Long-Context Fine-tuning); "with context window size varying from 1k to 64k in the dyadic fashion." (Section 4.3 Long-Context Evaluation)
> - Model sharing vs specialization: single LLaMA3 7B model pretrain -> fine-tune -> evaluate, "We pretrain Transformers with LLaMA3 7B architecture on 8k context length, fine-tune on 32k, and evaluate up to 64k." (Section 4 Experiments)
> - Role of positional encoding: central experimental variable with ablations, "We report the test perplexity for multiple Positional Encoding methods on context window sizes ranging from 1k to 64k on the checkpoints obtained from Subsection 4.2." (Section 4.4 Evaluation Results); "We compare two phase functions for TAPA: (i) *quadratic* (stationary) phase in (10) and (ii) *linear* (non-stationary) phase:" (Section 4.5 Ablations: TAPA's phase choice)

### 14. Final Classification

**Single-task, single-domain**

The evaluation is centered on language modeling over a single text dataset: "We evaluate all fine-tuned models on the test split of PG19 (Rae et al., 2019) which consist of mostly long sequence samples." (Section 4.3 Long-Context Evaluation) The paper frames the task as "long-context language modeling" (Section 3 A New Positional Encoding: Token-Aware Phase Attention (TAPA)) and does not present multi-domain or multi-task evaluations beyond this focus.
