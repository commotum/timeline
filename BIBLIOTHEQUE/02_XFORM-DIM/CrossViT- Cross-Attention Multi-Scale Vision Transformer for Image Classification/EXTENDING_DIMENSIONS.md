## 1. Basic Metadata

- Title: "CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification" (Title)
- Authors: "Chun-Fu (Richard) Chen, Quanfu Fan, Rameswar Panda MIT-IBM Watson AI Lab" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper proposes "a dual-branch transformer to combine image patches (i.e., tokens in a transformer) of different sizes to produce stronger image features" for "image classification" (Abstract).

---

## 3. Tasks Evaluated

- Task name: Image classification (ImageNet1K)
  - Task type: Classification
  - Dataset(s) used: ImageNet1K
  - Domain: Domain not specified.
  - Evidence: "We validate the effectiveness of our proposed approach on the ImageNet1K dataset [9]" (4.1. Experimental Setup)

- Task name: Image classification (CI-FAR10)
  - Task type: Classification
  - Dataset(s) used: CI-FAR10 [20]
  - Domain: Natural images
  - Evidence: "We validate this by performing transfer learning on 5 image classification tasks, including CI-FAR10 [20], CIFAR100 [20], Pet [27], CropDisease [23], and ChestXRay8 [40]. While the first four datasets contains natural images, ChestXRay8 consists of medical images." (Transfer Learning)

- Task name: Image classification (CIFAR100)
  - Task type: Classification
  - Dataset(s) used: CIFAR100 [20]
  - Domain: Natural images
  - Evidence: "We validate this by performing transfer learning on 5 image classification tasks, including CI-FAR10 [20], CIFAR100 [20], Pet [27], CropDisease [23], and ChestXRay8 [40]. While the first four datasets contains natural images, ChestXRay8 consists of medical images." (Transfer Learning)

- Task name: Image classification (Pet)
  - Task type: Classification
  - Dataset(s) used: Pet [27]
  - Domain: Natural images
  - Evidence: "We validate this by performing transfer learning on 5 image classification tasks, including CI-FAR10 [20], CIFAR100 [20], Pet [27], CropDisease [23], and ChestXRay8 [40]. While the first four datasets contains natural images, ChestXRay8 consists of medical images." (Transfer Learning)

- Task name: Image classification (CropDisease)
  - Task type: Classification
  - Dataset(s) used: CropDisease [23]
  - Domain: Natural images
  - Evidence: "We validate this by performing transfer learning on 5 image classification tasks, including CI-FAR10 [20], CIFAR100 [20], Pet [27], CropDisease [23], and ChestXRay8 [40]. While the first four datasets contains natural images, ChestXRay8 consists of medical images." (Transfer Learning)

- Task name: Image classification (ChestXRay8)
  - Task type: Classification
  - Dataset(s) used: ChestXRay8 [40]
  - Domain: Medical images
  - Evidence: "We validate this by performing transfer learning on 5 image classification tasks, including CI-FAR10 [20], CIFAR100 [20], Pet [27], CropDisease [23], and ChestXRay8 [40]. While the first four datasets contains natural images, ChestXRay8 consists of medical images." (Transfer Learning)

---

## 4. Domain and Modality Scope

- Single domain? No. Evidence: "While the first four datasets contains natural images, ChestXRay8 consists of medical images." (Transfer Learning)
- Multiple domains within the same modality? Yes (natural images and medical images within image modality). Evidence: "We validate this by performing transfer learning on 5 image classification tasks" and "While the first four datasets contains natural images, ChestXRay8 consists of medical images." (Transfer Learning)
- Multiple modalities? Not stated; only images are described (e.g., "image classification"). Evidence: "image classification" (Abstract)
- Does the paper claim domain generalization or cross-domain transfer? It claims transferability/generalization via transfer learning: "We also test the transferability of our approach using several smaller datasets" (4.1. Experimental Setup) and "it is crucial to check generalization of the models by evaluating transfer performance on tasks with fewer samples. We validate this by performing transfer learning on 5 image classification tasks" (Transfer Learning).

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image classification (ImageNet1K) | Not stated (single-task training on ImageNet1K) | No (trained on ImageNet1K) | Not specified. | "We validate the effectiveness of our proposed approach on the ImageNet1K dataset [9]" and "We train all our models for 300 epochs" (4.1. Experimental Setup) |
| Image classification (CI-FAR10) | Not stated (uses pretrained models per task) | Yes | Not specified. | "We finetune the whole pretrained models with 1,000 epochs" (Transfer Learning) |
| Image classification (CIFAR100) | Not stated (uses pretrained models per task) | Yes | Not specified. | "We finetune the whole pretrained models with 1,000 epochs" (Transfer Learning) |
| Image classification (Pet) | Not stated (uses pretrained models per task) | Yes | Not specified. | "We finetune the whole pretrained models with 1,000 epochs" (Transfer Learning) |
| Image classification (CropDisease) | Not stated (uses pretrained models per task) | Yes | Not specified. | "We finetune the whole pretrained models with 1,000 epochs" (Transfer Learning) |
| Image classification (ChestXRay8) | Not stated (uses pretrained models per task) | Yes | Not specified. | "We finetune the whole pretrained models with 1,000 epochs" (Transfer Learning) |

---

## 6. Input and Representation Constraints

- Fixed input resolution for evaluation: "During evaluation, we resize the shorter side of an image to 256 and take the center crop  $224 \times 224$  as the input." (4.1. Experimental Setup)
- Larger fixed resolution for some fine-tuning: "we also fine-tuned our models with a larger resolution ( $384 \times 384$ )" (4.1. Experimental Setup)
- Fixed patch size tokenization: "Vision Transformer (ViT) [11] first converts an image into a sequence of patch tokens by dividing it with a certain patch size" (3.1. Overview of Vision Transformer)
- Dual fixed patch sizes per branch: "two different branches to process image tokens of different sizes ( $P_s$  and  $P_l$ ,  $P_s < P_l$ )" (Figure 2)
- Fixed number of tokens per configuration: "N and C are the number of patch tokens and dimension of the embedding, respectively." (3.1. Overview of Vision Transformer)
- Image-only input assumption: "Vision Transformer (ViT) [11] first converts an image into a sequence of patch tokens" (3.1. Overview of Vision Transformer)
- Resizing/adjustment for embeddings: "Bicubic interpolation was applied to adjust the size of the learnt position embedding" (4.1. Experimental Setup)
- Token alignment via interpolation (within fusion): "We first perform an interpolation to align the spatial size" (3.3. Multi-Scale Feature Fusion)

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified. The sequence length is defined by the number of patch tokens N: "N and C are the number of patch tokens and dimension of the embedding, respectively." (3.1. Overview of Vision Transformer)
- Fixed or variable sequence length: Fixed per input resolution and patch size (implied by fixed patch size and input sizing). Evidence: "dividing it with a certain patch size" and "take the center crop  $224 \times 224$  as the input." (3.1. Overview of Vision Transformer; 4.1. Experimental Setup)
- Attention type: Global self-attention and cross-attention across branches. Evidence: "multiheaded self-attention (MSA)" (3.1. Overview of Vision Transformer) and "cross-attention module" where the CLS token interacts with patch tokens from the other branch (3.3. Multi-Scale Feature Fusion).
- Hierarchical/multi-scale structure: "Our architecture consists of a stack of K multi-scale transformer encoders" and uses two branches of different patch sizes (Figure 2).
- Computational cost controls: "our proposed cross-attention only requires linear time for both computational and memory complexity instead of quadratic time otherwise" and it "uses a single token for each branch as a query" (Abstract).

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Learnable position embedding added to tokens (absolute embedding via addition). Evidence: "ViT adds position embedding into each token, including the CLS token." (3.1. Overview of Vision Transformer)
- Where applied: At input, before transformer encoders. Evidence: "for each token of both branches, we also add a learnable position embedding before the multi-scale transformer encoder" (3.2. Proposed Multi-Scale Vision Transformer)
- Fixed or modified across experiments: Fixed in architecture, but resized for higher-resolution fine-tuning: "Bicubic interpolation was applied to adjust the size of the learnt position embedding" (4.1. Experimental Setup). No ablation or comparison is stated.

---

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption? Fixed architectural assumption; it is described as part of ViT/CrossViT input processing. Evidence: "ViT adds position embedding into each token" and "we also add a learnable position embedding before the multi-scale transformer encoder" (3.1. Overview of Vision Transformer; 3.2. Proposed Multi-Scale Vision Transformer).
- Are multiple positional encodings compared? Not stated.
- Any claim that PE choice is not critical or secondary? Not stated.

---

## 10. Evidence of Constraint Masking

- Model size/FLOPs tradeoffs: "our approach outperforms the recent DeiT by a large margin of 2% with a small to moderate increase in FLOPs and model parameters" (Abstract).
- Scaling model capacity: "CrossViT-9 $\dagger$  and CrossViT-15 $\dagger$  incur 30-50% more FLOPs and parameters than the baselines. However, their accuracy is considerably improved by  $\sim$  2.5-5%." (4.2. Main Results)
- Dataset size: "ImageNet1K contains 1,000 classes and the number of training and validation images are 1.28 millions and 50,000, respectively." (4.1. Experimental Setup)
- Architectural attribution: "This clearly demonstrates that our proposed cross-attention is effective in learning multiscale transformer features for image recognition." (4.2. Main Results)
- Training tricks: "These data augmentation methods include rand augmentation [8], mixup [47] and cutmix [46] as well as random erasing [49]." (4.1. Experimental Setup)

---

## 11. Architectural Workarounds

- Dual-branch multi-scale structure to manage complexity while leveraging multiple patch sizes: "we propose a dual-branch transformer to combine image patches (i.e., tokens in a transformer) of different sizes" and branches have "different computational complexity" (Abstract).
- Linear-time cross-attention fusion: "token fusion module based on cross attention, which uses a single token for each branch as a query" and "only requires linear time for both computational and memory complexity instead of quadratic time otherwise" (Abstract).
- CLS token as information exchange agent: "we first utilize the CLS token at each branch as an agent to exchange information among the patch tokens from the other branch" (3.3. Multi-Scale Feature Fusion).
- Balancing compute via uneven branch depth: "Our design includes different numbers of regular transformer encoders in the two branches (i.e. N and M) to balance computational costs." (Figure 2)
- Alternative patch tokenizers: "substituting the linear patch embedding in ViT by three convolutional layers as the patch tokenizer" (4.1. Experimental Setup).
- Avoiding extra FFN in cross-attention: "we do not apply a feed-forward network FFN after the cross-attention." (3.3. Multi-Scale Feature Fusion)

---

## 12. Explicit Limitations and Non-Claims

- Limitation/future work scope: "While our current work scratches the surface on multi-scale vision transformers for image classification, we anticipate that in future there will be more works in developing efficient multi-scale transformers for other vision applications, including object detection, semantic segmentation, and video action recognition." (5. Conclusion)
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Multiple image domains (natural and medical) within the same modality via transfer learning.
> - Task structure: Image classification only, evaluated across ImageNet1K and several transfer datasets.
> - Representation rigidity: Fixed patch sizes and fixed input resolutions per experiment (e.g., 224 x 224, 384 x 384).
> - Model sharing vs specialization: Pretrained models are fine-tuned per dataset; no joint multi-task training described.
> - Role of positional encoding: Learnable position embeddings added at input; not varied or compared.

---

### 14. Final Classification

**Multi-task, multi-domain (constrained).** The paper evaluates "transfer learning on 5 image classification tasks" and notes that "the first four datasets contains natural images, ChestXRay8 consists of medical images" (Transfer Learning), which indicates multiple domains within the image modality. The training setup is constrained to per-task fine-tuning of pretrained models ("We finetune the whole pretrained models with 1,000 epochs"), rather than joint multi-task training.
