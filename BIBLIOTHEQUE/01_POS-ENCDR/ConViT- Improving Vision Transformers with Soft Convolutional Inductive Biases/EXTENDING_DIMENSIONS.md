## 1. Basic Metadata

Title: "# **ConViT: Improving Vision Transformers** with Soft Convolutional Inductive Biases". (Section: Title)

Authors: "Stéphane d'Ascoli 12 Hugo Touvron 2 Matthew L. Leavitt 2 Ari S. Morcos 2 Giulio Biroli 12 Levent Sagun 2". (Section: Title)

Year: "Proceedings of the 38th International Conference on Machine Learning, PMLR 139, 2021." (Section: Front matter)

Venue: "Proceedings of the 38th International Conference on Machine Learning, PMLR 139, 2021." (Section: Front matter)

## 2. One-Sentence Contribution Summary

The paper introduces "gated positional self-attention (GPSA), a form of positional self-attention which can be equipped with a \"soft\" convolutional inductive bias" to answer whether it is possible "to combine the strengths of these two architectures while avoiding their respective limitations" (Section: Abstract).

## 3. Tasks Evaluated

Task 1: ImageNet-1k image classification.
Task type: Classification.
Dataset(s): ImageNet / ImageNet-1k.
Domain: Natural images.
Evidence: "Vision Transformers (ViTs) rely on more flexible self-attention layers, and have recently outperformed CNNs for image classification." (Section: Abstract) "The resulting convolutionallike ViT architecture, ConViT, outperforms the DeiT (Touvron et al., 2020) on ImageNet" (Section: Abstract) "Top-1 accuracy is measured on ImageNet-1k test set" (Section: Table 1 caption)

Task 2: Subsampled ImageNet-1k image classification.
Task type: Classification.
Dataset(s): Subsampled ImageNet-1k.
Domain: Natural images.
Evidence: "Both models are trained on a subsampled version of ImageNet-1k, where we only keep a variable fraction (leftmost column) of the images of each class for training." (Section: Table 2 caption) "we only keep a fraction  $f \in \{0.05, 0.1, 0.3, 0.5, 1\}$  of the images of each class." (Section: Figure 11 caption)

Task 3: CIFAR100 image classification.
Task type: Classification.
Dataset(s): CIFAR100.
Domain: Natural images.
Evidence: "Without any tuning, the ConViT also reaches high performance on CIFAR100" (Section: Performance of the ConViT) "For CIFAR100, we kept all hyperparameters unchanged, but rescaled the images to  $224 \times 224$  and increased the number of epochs" (Section: C. Further performance results)

Task 4: ImageNet (first 100 classes) image classification.
Task type: Classification.
Dataset(s): ImageNet (first 100 classes).
Domain: Natural images.
Evidence: "We examined the effects of these hyperparameters on ConViT-S, trained on the first 100 classes of ImageNet." (Section: 4. Investigating the role of locality) "we train the ConViT-S+ for 300 epochs on first 100 classes of ImageNet." (Section: Figure 12 caption)

## 4. Domain and Modality Scope

Evaluation is performed on multiple datasets within the same modality (natural images), not multiple modalities. Evidence: "The resulting convolutionallike ViT architecture, ConViT, outperforms the DeiT (Touvron et al., 2020) on ImageNet" (Section: Abstract) and "For CIFAR100, we kept all hyperparameters unchanged, but rescaled the images to  $224 \times 224$" (Section: C. Further performance results).

Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ImageNet-1k image classification | Not specified; trained per task. | Not specified. | Not specified. | "trained from scratch on ImageNet." (Section: Table 1 caption) |
| Subsampled ImageNet-1k image classification | Not specified; trained per task. | Not specified. | Not specified. | "Both models are trained on a subsampled version of ImageNet-1k" (Section: Table 2 caption) |
| CIFAR100 image classification | Not specified; trained per task. | Not specified. | Not specified. | "For CIFAR100, we kept all hyperparameters unchanged, but rescaled the images to  $224 \times 224$  and increased the number of epochs" (Section: C. Further performance results) |
| ImageNet (first 100 classes) image classification | Not specified; trained per task. | Not specified. | Not specified. | "we train the ConViT-S+ for 300 epochs on first 100 classes of ImageNet." (Section: Figure 12 caption) |

## 6. Input and Representation Constraints

- Fixed input resolution and patch grid: "The ViT slices input images of size 224 into  $16 \times 16$  non-overlapping patches of  $14 \times 14$  pixels" (Section: Architectural details).
- Fixed patch size: "embeddings of  $16 \times 16$  pixel patches" (Section: 2. Background).
- Sequence of patch embeddings: "a sequence of L embeddings" (Section: 2. Background).
- Embedding dimensionality tied to heads: "embeds them into vectors of dimension  $D_{\rm emb} = 64 N_h$" (Section: Architectural details).
- 2D relative position assumption: "the relative positional encodings  $r_{ij} \in \mathbb{R}^{D_{\text{pos}}}$  only depend on the distance between pixels i and j, denoted denoted by a two-dimensional vector  $\boldsymbol{\delta}_{ij}$ ." (Section: 2. Background).
- Resizing requirement for CIFAR100: "rescaled the images to  $224 \times 224$" (Section: C. Further performance results).
- Positional-embedding interpolation when changing resolution: "dispensing with the need to interpolate the embeddings when changing the input resolution" (Section: Architectural details).

## 7. Context Window and Attention Structure

Maximum sequence length: Not specified; the paper refers to "a sequence of L embeddings" without stating a numeric L. (Section: 2. Background)

Sequence length fixed or variable: Fixed for the stated 224-sized inputs and 16x16 patch grid (implying a fixed sequence length per input size), with changes requiring interpolation/resampling. Evidence: "input images of size 224 into  $16 \times 16$  non-overlapping patches" and "dispensing with the need to interpolate the embeddings when changing the input resolution" (Section: Architectural details).

Attention type: Global self-attention over all patches with positional attention in GPSA. Evidence: "performing SA across embeddings of patches of pixels" (Section: 1. Introduction) and "replace the vanilla SA with positional self-attention (PSA), using encodings  $r_{ij}$  of the relative position of patches i and j" (Section: 2. Background).

Mechanisms to manage computational cost: The paper reduces positional-parameter count by fixing relative encodings. Evidence: "the number of relative positional encodings  $r_{\delta}$  is quadratic in the number of patches" and "we leave the relative positional encodings  $r_{\delta}$  fixed, and train only the embeddings  $v_{pos}^h$" (Section: 3. Approach).

## 8. Positional Encoding (Critical Section)

Mechanism and placement:
- Absolute positional embeddings at input: "the positional information is instead injected to each patch before the first layer, by adding a learnable positional embedding of dimension  $D_{\rm emb}$" (Section: Architectural details).
- Relative positional encodings in attention: "replace the vanilla SA with positional self-attention (PSA), using encodings  $r_{ij}$  of the relative position of patches i and j" (Section: 2. Background), and GPSA uses "relative position encodings (fixed)" (Section: Figure 4 caption).
- GPSA combines content and positional attention with gating: "GPSA layers sum the content and positional terms *after* the softmax, with their relative importances governed by a learnable *gating* parameter  $\lambda_h$" (Section: 3. Approach).

Fixed vs modified across experiments:
- Absolute positional embeddings kept: "we keep the absolute positional embeddings of the ViT active in the ConViT" (Section: Architectural details).
- Relative positional encodings fixed: "we leave the relative positional encodings  $r_{\delta}$  fixed" (Section: 3. Approach).
- Ablation of absolute embeddings: "In Tab. 5, we explore the importance of the absolute positional embeddings injected to the input in both the DeiT and ConViT." (Section: F. Further ablations).

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption? The main architecture keeps absolute and relative positional encodings, but positional encoding is explicitly ablated in experiments. Evidence: "we keep the absolute positional embeddings of the ViT active in the ConViT" (Section: Architectural details) and "In Tab. 5, we explore the importance of the absolute positional embeddings" (Section: F. Further ablations).
- Multiple positional encodings compared? The paper tests masking of absolute positional embeddings rather than comparing multiple PE mechanisms. Evidence: "masking them off at test time" (Section: F. Further ablations).
- Any claim that PE is not critical? The paper states the absolute PE is less important: "This also shows that the absolute positional information contained in the embeddings is not very useful." (Section: F. Further ablations).

## 10. Evidence of Constraint Masking

- Model sizes: Table 1 lists multiple parameter scales, e.g., "6M" (Ti), "22M" (S), "86M" (B), and "152M" (B+) (Section: Table 1).
- Dataset size scaling: "we only keep a fraction  $f \in \{0.05, 0.1, 0.3, 0.5, 1\}$  of the images of each class" (Section: Figure 11 caption) and "Both models are trained on a subsampled version of ImageNet-1k" (Section: Table 2 caption).
- Gains attributed to architectural locality: "The convolutional inductive bias strongly improves sample efficiency." (Section: Table 2 caption) and "Strong locality is desirable" (Section: 4. Investigating the role of locality).
- Scaling/model size effects: "The relative improvement of the ConViT over the DeiT increases with model size." (Section: Figure 11 caption).
- Training tricks: "The hard distillation introduced in Touvron et al. (2020) greatly improves the performance of the DeiT." (Section: B. The effect of distillation).

## 11. Architectural Workarounds

- GPSA layers to inject soft convolutional bias: "replacing some of the SA layers by a new type of layer which we call *gated positional self-attention* (GPSA) layers" (Section: 3. Approach).
- Convolutional initialization to impose locality early: "We initialize the GPSA layers to mimic the locality of convolutional layers" (Section: Abstract).
- Adaptive attention span via fixed relative encodings: "we leave the relative positional encodings  $r_{\delta}$  fixed, and train only the embeddings  $v_{pos}^h$" (Section: 3. Approach).
- Positional gating to balance content vs position: "GPSA layers sum the content and positional terms *after* the softmax, with their relative importances governed by a learnable *gating* parameter" (Section: 3. Approach).
- Class token placement workaround: "We solve this problem by appending the class token to the patches after the last GPSA layer" (Section: Architectural details).

## 12. Explicit Limitations and Non-Claims

- Future work direction: "Another direction which will be explored in future work is the following: if SA layers benefit from being initialized as random convolutions, could one reduce even more drastically their sample complexity by initializing them as pre-trained convolutions?" (Section: 5. Conclusion and perspectives).
- Open question about attention necessity: "This naturally begs the question: is attention really key to the success of ViTs (Dong et al., 2021; Tolstikhin et al., 2021; Touvron et al., 2021a)?" (Section: 4. Investigating the role of locality).
- No explicit statements about open-world learning, unrestrained multi-task learning, or meta-learning. (Not specified.)

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Multiple datasets within a single modality (natural images: ImageNet/CIFAR100), no cross-modal evaluation.
> - Task structure: Image classification across datasets/subsets; no evidence of joint multi-task training.
> - Representation rigidity: Fixed 224-sized inputs with 16x16 patch grid and fixed patch embeddings; 2D positional assumptions.
> - Model sharing vs specialization: Separate training runs per dataset/subset; no explicit shared-weight multitask setup.
> - Role of positional encoding: Absolute input embeddings plus fixed relative encodings in GPSA; absolute PE is ablated but retained in main model.

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple datasets and subsets within the same image modality, e.g., "ImageNet" and "CIFAR100" (Sections: Abstract; C. Further performance results), but all tasks are image classification and there is no evidence of joint multi-task training. The evidence points to multiple classification evaluations within a single domain rather than unrestrained multi-domain learning.
