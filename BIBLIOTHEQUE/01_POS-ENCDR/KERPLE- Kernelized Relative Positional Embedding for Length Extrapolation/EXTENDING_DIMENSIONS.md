## 1. Basic Metadata

- Title: KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation. Evidence: "# **KERPLE: Kernelized Relative Positional Embedding** for Length Extrapolation" (Title block)
- Authors: Ta-Chung Chi*; Peter J. Ramadge; Ting-Han Fan*; Alexander I. Rudnicky. Evidence: "Ta-Chung Chi*" (Title block); "Peter J. Ramadge" (Title block); "Ting-Han Fan*" (Title block); "Alexander I. Rudnicky" (Title block)
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

"We propose KERPLE, a framework that generalizes relative position embedding for extrapolation by kernelizing positional differences." (Abstract)

## 3. Tasks Evaluated

Task 1
- Task name: Language modeling (length extrapolation) on OpenWebText2
- Task type: Generation
- Dataset(s) used: OpenWebText2
- Domain: natural language web text (internet)
- Evidence: "Experiments demonstrate that the logarithmic variant achieves excellent extrapolation performance on three large language modeling datasets." (Abstract) "We conduct experiments on OpenWebText2, GitHub, and ArXiv datasets gathered in Gao et al. [2020]." (Section 5.1 Dataset and Implementation Description) "OpenWebText2 includes recent content from Reddit submissions until 2020, content from multiple languages, document metadata, multiple dataset versions, and open-source replication code." (Section 5.1 Dataset and Implementation Description)

Task 2
- Task name: Language modeling (length extrapolation) on GitHub
- Task type: Generation
- Dataset(s) used: GitHub
- Domain: programming language text (code)
- Evidence: "Experiments demonstrate that the logarithmic variant achieves excellent extrapolation performance on three large language modeling datasets." (Abstract) "We conduct experiments on OpenWebText2, GitHub, and ArXiv datasets gathered in Gao et al. [2020]." (Section 5.1 Dataset and Implementation Description) "GitHub includes open-source repositories written in primary coding languages such as Java, C/C++, Python, and Go." (Section 5.1 Dataset and Implementation Description)

Task 3
- Task name: Language modeling (length extrapolation) on ArXiv
- Task type: Generation
- Dataset(s) used: ArXiv
- Domain: academic paper text (scientific articles)
- Evidence: "Experiments demonstrate that the logarithmic variant achieves excellent extrapolation performance on three large language modeling datasets." (Abstract) "We conduct experiments on OpenWebText2, GitHub, and ArXiv datasets gathered in Gao et al. [2020]." (Section 5.1 Dataset and Implementation Description) "ArXiv includes papers written in LaTex in Math, Computer Science, Physics, and some related fields." (Section 5.1 Dataset and Implementation Description)

## 4. Domain and Modality Scope

- Evaluation performed on multiple domains within the same modality (text). Evidence: "To evaluate the applicability of the model in different domains, we conduct experiments on OpenWebText2, GitHub, and ArXiv datasets." (Section 5.2 Experimental Results)
- Multiple modalities? Not stated; datasets described are text corpora. Evidence: "OpenWebText2 includes recent content from Reddit submissions until 2020, content from multiple languages, document metadata, multiple dataset versions, and open-source replication code. GitHub includes open-source repositories written in primary coding languages such as Java, C/C++, Python, and Go. ArXiv includes papers written in LaTex in Math, Computer Science, Physics, and some related fields." (Section 5.1 Dataset and Implementation Description)
- Domain generalization or cross-domain transfer claims: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| OpenWebText2 language modeling | Not specified. | Not specified. | Not specified. | "We conduct experiments on OpenWebText2, GitHub, and ArXiv datasets gathered in Gao et al. [2020]." (Section 5.1 Dataset and Implementation Description) |
| GitHub language modeling | Not specified. | Not specified. | Not specified. | "We conduct experiments on OpenWebText2, GitHub, and ArXiv datasets gathered in Gao et al. [2020]." (Section 5.1 Dataset and Implementation Description) |
| ArXiv language modeling | Not specified. | Not specified. | Not specified. | "We conduct experiments on OpenWebText2, GitHub, and ArXiv datasets gathered in Gao et al. [2020]." (Section 5.1 Dataset and Implementation Description) |

## 6. Input and Representation Constraints

- Tokenized sequence input with variable length L and fixed embedding dimension d: "Let  $\{w_m\}_{m=1}^L$  be the input tokens to a transformer model, where L is the total number of tokens. Each  $w_m$  is a scalar and is used to index the embedding vector  $\boldsymbol{e}_m \in \mathbb{R}^d$  as the input to the transformer." (Section 2.1 Preliminary)
- Fixed training sequence length and max positions: "We adopt almost all configurations of small GPT-NeoX, except that we change the train-micro-batch-size to 32, seq-length to 512, and max-position-embeddings to 512." (Section 5.1 Dataset and Implementation Description)
- Training length vs evaluation length range: "we train our model with length 512 and test on lengths ranging from 512 to 16384." (Section 5.2 Experimental Results)
- Fixed hidden size (embedding dimensionality): "Hidden size 64 means that d = 64 in Eq. (1)." (Section 5.1 Dataset and Implementation Description)
- Fixed patch size: Not specified.
- Fixed number of tokens: Training uses fixed seq-length 512; evaluation varies. Evidence: "We adopt almost all configurations of small GPT-NeoX, except that we change the train-micro-batch-size to 32, seq-length to 512, and max-position-embeddings to 512." (Section 5.1 Dataset and Implementation Description) "we train our model with length 512 and test on lengths ranging from 512 to 16384." (Section 5.2 Experimental Results)
- Fixed dimensionality (e.g., strictly 2D): Not specified beyond token embedding dimension d.
- Padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length evaluated: 16384. Evidence: "we train our model with length 512 and test on lengths ranging from 512 to 16384." (Section 5.2 Experimental Results)
- Sequence length fixed or variable: Training fixed at 512; evaluation varies. Evidence: "We adopt almost all configurations of small GPT-NeoX, except that we change the train-micro-batch-size to 32, seq-length to 512, and max-position-embeddings to 512." (Section 5.1 Dataset and Implementation Description) "we train our model with length 512 and test on lengths ranging from 512 to 16384." (Section 5.2 Experimental Results)
- Attention type: Global (full) self-attention over all tokens. Evidence: "a_{m,n} = \frac{\exp(\boldsymbol{q}_m^{\top} \boldsymbol{k}_n / \sqrt{d})}{\sum_{i=1}^{L} \exp(\boldsymbol{q}_m^{\top} \boldsymbol{k}_i / \sqrt{d})}, \quad \boldsymbol{o}_m = \sum_{n=1}^{L} a_{m,n} \boldsymbol{v}_n." (Section 2.1 Preliminary)
- Mechanisms to manage computational cost: Train short, test long to avoid larger L during training. Evidence: "Training (or retraining) the model using a substantially larger value of L is often infeasible since the transformer training cost is  $O(L^2)$ ." (Section 1 Introduction) "we train our model with length 512 and test on lengths ranging from 512 to 16384." (Section 5.2 Experimental Results)

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Relative positional embedding via kernelized bias added to attention scores. Evidence: "Relative positional embeddings (RPE) encode the idea of shift-invariance: for any shift p, (m+p)-(n+p)=m-n. It is often added directly to the self-attention matrix before Softmax normalization." (Section 1 Introduction) "We propose a kernelized relative positional embedding as follows." (Section 4 Kernelized Relative Positional Embedding)
- Where applied: Attention bias added before Softmax. Evidence: "It is often added directly to the self-attention matrix before Softmax normalization." (Section 1 Introduction) "a_{m,n} = \frac{\exp\left((\mathbf{q}_{m}^{\top} \mathbf{k}_{n} + \tilde{k}_{r_{1},\dots,r_{\ell}}(m,n))/\sqrt{d}\right)}{\sum_{i=1}^{L} \exp\left((\mathbf{q}_{m}^{\top} \mathbf{k}_{i} + \tilde{k}_{r_{1},\dots,r_{\ell}}(m,i))/\sqrt{d}\right)}" (Section 4 Kernelized Relative Positional Embedding)
- Applied per head and shared across layers: "a, b, and p are learnable parameters in each attention head shared across layers." (Figure 1 caption)
- Positional encoding fixed or modified across experiments: Multiple KERPLE variants and baselines are compared. Evidence: "we fix  $\ell=2$  and experiment on two variants of the composite kernel, Eq. (4), where we call these the power variant and the logarithmic variant of our proposed KERPLE framework" (Section 4 Kernelized Relative Positional Embedding) "we compare KERPLE with Sinusoidal [Vaswani et al., 2017], Rotary [Su et al., 2021], T5 [Raffel et al., 2020], and ALiBi [Press et al., 2022]." (Section 5.2 Experimental Results)

## 9. Positional Encoding as a Variable

- Positional encoding is a core research variable: "Our main result is a framework for **KE**rnelize **R**elative **P**ositional Embedding for **L**ength **E**xtrapolation (**KERPLE**)." (Section 1 Introduction)
- Multiple positional encodings compared: "we compare KERPLE with Sinusoidal [Vaswani et al., 2017], Rotary [Su et al., 2021], T5 [Raffel et al., 2020], and ALiBi [Press et al., 2022]." (Section 5.2 Experimental Results)
- PE choice claimed not critical or secondary: Not stated.

## 10. Evidence of Constraint Masking

- Model size: "Table 2: 162M Model Configurations." (Section 5.1 Dataset and Implementation Description) "| 12             | 64          | 12                | 512             | 162M                         |" (Section 5.1 Dataset and Implementation Description)
- Dataset sizes: "Table 1: **Dataset Overview.** Raw Size is the size before any up- or down-sampling." (Section 5.1 Dataset and Implementation Description) "| Raw Size | 66.77 GB     | 95.16 GB | 56.21 GB |" (Section 5.1 Dataset and Implementation Description)
- Performance gains attributed to architectural choices (KERPLE variants), not scaling: "Experiments demonstrate that the logarithmic variant achieves excellent extrapolation performance on three large language modeling datasets." (Abstract) "the logarithmic variant consistently outperforms prior work at all extrapolation lengths and tasks." (Section 5.2 Experimental Results)
- Configurations held fixed across experiments: "Table 2 summarizes the important configurations fixed throughout our experiments." (Section 5.1 Dataset and Implementation Description)
- Scaling model size or data as primary driver: Not stated.

## 11. Architectural Workarounds

- Window attention appears only as a baseline, not as the core architecture: "KERPLE-log-windowed@512" (Section 5.5 Position-wise Perplexity Evaluation) "Although window attention is a strong baseline, our KERPLE-log is almost like a free lunch compared to window attention" (Section 5.5 Position-wise Perplexity Evaluation)
- Other architectural scale-management techniques (hierarchical stages, token pooling, task-specific heads, fixed grids): Not stated.

## 12. Explicit Limitations and Non-Claims

- Stated limitations / future work: "We believe our work paves the way for some interesting future directions that resolve our limitations. For instance, we can consider general kernel families and model non-monotonic effects due to positional differences. In addition, the use of learnable parameters in KERPLE might enable better generalization to inputs higher than one-dimensional. Last but not least, there is always room for improving memory efficiency by adjusting the model architecture and training procedure." (Section 6 Conclusion and Future Work)
- Explicit non-claims about open-world, unrestrained multi-task, or meta-learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: multiple text domains (internet, code, academic) within one modality.
> - Task structure: language modeling / length extrapolation on three datasets; no other task types evaluated.
> - Representation rigidity: training seq length fixed at 512 with max-position-embeddings 512; evaluation varies up to 16384.
> - Model sharing vs specialization: per-dataset weight sharing or joint training not specified.
> - Role of positional encoding: central research variable with multiple PE variants compared.

### 14. Final Classification

**Multi-task, multi-domain (constrained).** The evaluation spans three distinct text domains: "We conduct experiments on OpenWebText2, GitHub, and ArXiv datasets gathered in Gao et al. [2020]." (Section 5.1 Dataset and Implementation Description), and the task is language modeling on these datasets ("three large language modeling datasets" (Abstract)). The setup remains constrained to a single modality and a single task type, with no evidence of unrestrained multi-task training.
