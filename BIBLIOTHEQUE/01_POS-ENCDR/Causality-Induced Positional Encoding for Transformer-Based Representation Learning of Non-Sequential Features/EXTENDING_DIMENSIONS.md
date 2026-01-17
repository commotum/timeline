## 1. Basic Metadata

- Title: "Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features" (Title)
- Authors: "Kaichen Xu<sup>1,2</sup>, Yihang Du<sup>1</sup>, Mianpeng Liu<sup>1</sup>, Zimu Yu<sup>1</sup>, Xiaobo Sun<sup>1,3\*</sup>" (Author list)
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

- "To address this limitation, we propose CAPE, a novel method that identifies underlying causal structure over non-sequential features as a weighted directed acyclic graph (DAG) using generalized structural equation modeling." (Abstract)

## 3. Tasks Evaluated

- Task name: Synthetic causal structure identification / CAPE property evaluation. Task type: Other (causal structure recovery). Dataset(s) used: "we simulate a tabular dataset  $X_{\rm syn} \in \mathbb{R}^{5000 \times 10}$ , consisting of 5,000 observations over a set  $\mathcal V$  of ten non-sequential features." (Section 5.1). Domain: synthetic tabular features. Evidence: "CAPE effectively identifies the causal structure and preserves it in the hyperbolic manifold." (Section 5.1)
- Task name: Gene Perturbation Prediction (GPP). Task type: Other (perturbation effect prediction). Dataset(s) used: "The Norman perturbation dataset provides gene expression profiles from the K562 leukemia cell line treated with Perturb-seq. This dataset includes 131 dual-gene perturbations and 105 single-gene perturbations, with each perturbation represented by approximately 300 to 700 cells." (Section D.2.1). Domain: single-cell gene expression (scRNA-seq). Evidence: "GPP aims to leverage the learned gene representations to predict perturbation (e.g., gene knockout or activation)-induced changes in gene expression profiles, facilitating the exploration of gene functions and regulatory networks." (Section 5.2)
- Task name: Cell type annotation. Task type: Classification. Dataset(s) used: "cell embeddings are learned using scGPT and scBERT with three types of positional encoding across three human datasets (hPBMC, hPancreas, and hBMMC) and one mouse dataset (mOP) (See Section D.2 for details)." (Section G.2). Domain: single-cell transcriptomics and spatial transcriptomics, e.g., "The hPBMC [80] dataset, sourced from a healthy donor, contains gene expression profiles for 68,450 peripheral blood mononuclear cells (PBMCs). These cells were processed using the 10x platform with scRNA-seq technology." and "It was generated using Multiplexed Error-Robust Fluorescence In Situ Hybridization (MER-FISH), a spatial transcriptomic technique that enables gene expression profiling while preserving the spatial organization of cells within tissue sections." (Section D.2.2). Evidence: "As a standard classification task, we adopt the evaluation framework established in prior studies [22–25, 87]." (Section E.2)
- Task name: Cell clustering. Task type: Other (clustering). Dataset(s) used: "For single cell proteomics, we evaluate the cell embeddings in the cell clustering task, applying it to two datasets: SCoPE2\_Specht and SCoPE2\_Montalvo (see Section D.2 for details)." (Section G.2). Domain: single-cell proteomics, e.g., "SCoPE2\_Specht [45] is a representative single-cell proteomic dataset that quantifies 3,042 proteins in 1,490 cells using the SCoPE2 method." (Section D.2.3). Evidence: "we conducted a cell clustering experiment, which is a standard practice in single-cell proteomics [88, 89]." (Section E.3)
- Task name: Age prediction. Task type: Other (age regression). Dataset(s) used: "We use a widely used DNA methylation dataset for age prediction, collected by [86], which includes 13,505 samples (21,368 CpG sites) from multiple tissues." (Section D.2.4). Domain: DNA methylation (epigenomics), "We further assess CAPE's performance in predicting age from DNA methylation patterns." (Section G.2). Evidence: "Following established DNA methylation foundational models [77], we fine-tuned both our model and MethylGPT using a ResNet1D prediction head." (Section E.4)

## 4. Domain and Modality Scope

- Single domain? No. The paper evaluates "data from multiple omics domains [43], including transcriptomics, epigenomics, and proteomics" (Section 5.2).
- Multiple domains within the same modality? Yes; the evaluation spans "multiple omics domains [43], including transcriptomics, epigenomics, and proteomics" (Section 5.2).
- Multiple modalities? Multiple omics modalities are included: "transcriptomics, epigenomics, and proteomics" (Section 5.2).
- Does the paper claim domain generalization or cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Synthetic causal structure identification | Not applicable | Not applicable | Not applicable | "CAPE is trained to estimate A from  $X_{\mathrm{syn}}$ ." (Section 5.1) |
| Gene Perturbation Prediction (GPP) | Not specified (trained for this task) | Not specified | Yes (GEARS) | "the two methods are trained on unperturbed cells to learn contextualized gene representations, which are fed into GEARS [44], a perturbation prediction model" (Section 5.2) |
| Cell type annotation | No (task-specific fine-tuning) | Yes | Yes (classifier) | "Under the fine-tuning setting, we append an additional classifier to the cell embeddings generated by each model and perform supervised fine-tuning on the model parameters to optimize task-specific performance." (Section E.2) |
| Cell clustering | No (task/domain-specific fine-tuning) | Yes | No (k-means) | "we fine-tuned it on the pSCoPE\_Leduc dataset to adapt it for proteomics data." (Section G.2) and "We employed the k-means algorithm to obtain the cell clusters" (Section E.3) |
| Age prediction | No (task-specific fine-tuning) | Yes | Yes (ResNet1D) | "we fine-tuned both our model and MethylGPT using a ResNet1D prediction head. During joint optimization, both the pre-trained MethylGPT and the downstream ResNet1D were trained end-to-end" (Section E.4) |

## 6. Input and Representation Constraints

- Fixed number of tokens/features: "Let  $\mathcal{V} = \{v_j\}_{j=1}^M$  be a sequence of M input tokens" and "we assume that  $\{v_j\}_{j=1}^M$  are causally related and organized into a tabular measurement dataset  $X \in \mathbb{R}^{N \times M}$" (Section 3.1).
- Gene list normalization with zero fill: "unmapped genes are assigned zero expression values, thereby enforcing uniform gene symbol compatibility across all processed matrices." (Section F.1.1)
- Discretization/binning constraint: "it calculates the raw absolute values and divide them into B consecutive intervals  $[b_k, b_{k+1}]$" (Section F.2)
- Spatial coordinates ignored in preprocessing: "spatial transcriptomics data (e.g., Slide-seq) is processed consistently and does not take into account spatial coordinates and H&E images." (Section D.1.1)
- Positional encoding dimensionality: "d = D/2, and the dimensionality of variable embeddings D is determined by the selected transformer backbones (e.g., D = 200 for scBERT and D = 512 for scGPT)." (Section F.4)
- Fixed/variable input resolution, fixed patch size, padding/resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Sequence length: "Let  $\mathcal{V} = \{v_j\}_{j=1}^M$  be a sequence of M input tokens" and "at the beginning of the input sequence  $v_1^i, v_2^i, \cdots, v_M^i$  of cell i, scBERT sets a special <cls> token" (Sections 3.1, F.2).
- Maximum sequence length / fixed vs. variable: Not specified beyond the use of M tokens in the sequence (Section 3.1).
- Attention type: described as self-attention, e.g., "self-attention mechanism" (Abstract) and scBERT uses a "transformer backbone (Performer [96])" (Section F.2).
- Computational cost mechanisms: CAPE's rotary form is noted as "compatibility with linear self-attention" (Section 1 Introduction).

## 8. Positional Encoding (Critical Section)

- Mechanism: CAPE produces causality-aware positional encodings and converts them to rotary form, e.g., "This step yields causality-aware positional encodings for the features, which are converted into their rotary form for integrating with transformer's self-attention mechanism." (Abstract) and "CAPE converts the hyperbolic positional encodings into rotary form, a causality-induced version of RoPE [18]." (Section 1 Introduction)
- Where applied: "In Step III (Section 3.5), hyperboloid positional embeddings are mapped into a unit Poincaré ball via diffeomorphism before being transformed into their rotary form for modulating feature-wise attention scores in the transformer." (Section 3.2) and "Following RoPE, positional encodings are only injected into keys and queries, not values." (Section 3.5)
- Fixed vs. modified across experiments: positional encodings are varied in evaluation, e.g., "Different position encoding approaches, which do not rely on predefined feature order <sup>7</sup>, are evaluated with the two models, including CAPE, their default methods, and a trainable, causality-agnostic relative position encoder [49]." (Section 5.2)
- Ablations and alternatives: "In these experiments, we adopt scGPT as the transformer backbone, replacing its default position encoding mechanism with four CAPE ablation variants." and "Lastly, the fourth variant (CAPE-w/o-rotary) bypasses the rotary form conversion (Step III), directly adding hyperbolic positional encodings to feature embeddings." (Section 5.3)

## 9. Positional Encoding as a Variable

- Core research variable? Yes; "Different position encoding approaches, which do not rely on predefined feature order <sup>7</sup>, are evaluated with the two models, including CAPE, their default methods, and a trainable, causality-agnostic relative position encoder [49]." (Section 5.2)
- Multiple positional encodings compared? Yes; "Different position encoding approaches, which do not rely on predefined feature order <sup>7</sup>, are evaluated with the two models, including CAPE, their default methods, and a trainable, causality-agnostic relative position encoder [49]." (Section 5.2)
- PE choice claimed as not critical or secondary? Not claimed.

## 10. Evidence of Constraint Masking

- Model size (embedding dimensions): "d = D/2, and the dimensionality of variable embeddings D is determined by the selected transformer backbones (e.g., D = 200 for scBERT and D = 512 for scGPT)." (Section F.4)
- Model size (architecture depth/heads): "The backbone network has 4 transformer blocks, each with 8 attention heads." (Section F.4)
- Dataset scale (single-cell pretraining): "This collection includes 1,465 datasets, encompassing around 91.5 million cells and covering approximately 900 different cell types, with data spanning several sequencing methods and omics modalities." (Section D.1.1)
- Dataset scale (DNA methylation pretraining): "We adopt the pretraining dataset released by MethylGPT [77], which consists of DNA methylation data collected from 154,063 human samples through the EWAS Data Hub [78] and Clockbase [79]." and "We specifically focus on 49,156 CpG sites selected for their biological relevance and array format compatibility, as detailed by the EWAS catalog." (Section D.1.2)
- Performance attribution: improvements are tied to positional encoding choices, e.g., "We find that both models equipped with CAPE consistently yield substantial performance gains (11.1% average reduction in MSE) compared to their respective default approaches." (Section 5.2). No explicit claim that gains are primarily due to scaling model size or data.

## 11. Architectural Workarounds

- Low-rank approximation for scalability: "To mitigate this computational bottleneck, we adopt the low-rank approximation strategy proposed by Dong, et al. [107] in our implementation. Specifically, A is approximated as  $UV^{\top}$  with  $U, V \in \mathbb{R}^{M \times r}$  and rank r = 40. This approximation reduces the computation complexity of acyclicity constraint to  $\mathcal{O}(M^2r)$ , yielding an overall complexity of  $\mathcal{O}((N+r)M^2)$  for Step I." (Section G.4)
- Efficient attention backbone: scBERT uses a "transformer backbone (Performer [96])" and CAPE's rotary form emphasizes "compatibility with linear self-attention" (Sections F.2, 1 Introduction).
- Token pooling for observation-level embeddings: "Agg denotes an aggregate function (e.g., mean or max pooling)" (Section 3.1).
- Task-specific heads: "fed into GEARS [44], a perturbation prediction model" (Section 5.2), "append an additional classifier" (Section E.2), and "ResNet1D prediction head" (Section E.4).

## 12. Explicit Limitations and Non-Claims

- "its effectiveness currently relies on the quality of the inferred causal graph. Although we adopt a robust variational formulation for causal discovery, inaccuracies may arise in extremely noisy or undersampled settings." (Section H)
- "our current implementation assumes feature-wise causal structure to be static across samples, which may not fully capture sample-specific heterogeneity in highly dynamic systems. These limitations point to promising directions for future work, such as incorporating uncertainty-aware causal discovery or adapting CAPE to sample-dependent causal structures." (Section H)

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Multiple omics domains (transcriptomics, epigenomics, proteomics), not a single domain.
> - Task structure: Multiple downstream tasks (GPP, cell type annotation, clustering, age prediction) defined within single-cell/omics settings.
> - Representation rigidity: Tabular X in R^{N x M}, fixed feature sets, binning and zero-fill preprocessing, and spatial coordinates ignored for spatial transcriptomics.
> - Model sharing vs specialization: Models are fine-tuned per task with task-specific heads (classifier, GEARS, ResNet1D).
> - Role of positional encoding: Central research variable with explicit comparisons and ablations.

### 14. Final Classification

**Multi-task, multi-domain (constrained)**

The paper evaluates "data from multiple omics domains [43], including transcriptomics, epigenomics, and proteomics, (see Section D for the data description)." (Section 5.2) It also states: "Feature and observation representations generated by the CAPE-transformer model are evaluated in various feature-level and observation-level downstream tasks. Here, we focus on the feature-level task, gene perturbation prediction (GPP) with scRNA-seq data [44], and leave the results of other tasks, e.g., cell clustering with proteomics data [45] and age prediction with epigenomics data [46], to Section G.2." (Section 5.2) These are defined omics tasks/datasets, indicating a constrained multi-task, multi-domain setup rather than unrestrained multi-domain learning.
