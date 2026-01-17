## 1. Basic Metadata

- Title: "LieRE: Lie Rotational Positional Encodings" (Title)
- Authors: "Sophie Ostmeier 1\* Brian Axelrod \* Maya Varma 1 Michael Moseley 2 Akshay Chaudhari 2† Curtis Langlotz 2†" (Title page)
- Year: "PMLR 267, 2025." (front matter)
- Venue (conference/journal/arXiv): "Proceedings of the 42nd International Conference on Machine" and "Learning, Vancouver, Canada. PMLR 267, 2025." (front matter)

## 2. One-Sentence Contribution Summary

The paper claims to "introduce Lie Relative Encodings (LieRE)" as a "principled generalization of RoPE" to address limitations for "modalities with high dimensional structure" (Abstract).

## 3. Tasks Evaluated

- Task name: 2D image classification; Task type: Classification; Dataset(s): CIFAR-100, ImageNet-1k; Domain: natural images; Evidence: "We begin with CIFAR-100 and ImageNet-1k benchmarks to evaluate LieRE in 2D vision tasks." (Section 5.1. 2D Image Classification)
- Task name: Synthetic spatial reasoning (arrow direction) image classification; Task type: Classification; Dataset(s): synthetic task; Domain: synthetic grid images; Evidence: "we designed a synthetic image classification task (Shah et al., 2024)." and "The task presents a  $108 \times 108$  pixel image containing a  $9 \times 9$  grid (81 cells)." and "The objective is to identify the direction of this specific arrow." (Section 5.2. Synthetic Spatial Reasoning Task)
- Task name: 3D video classification; Task type: Classification; Dataset(s): UCF101; Domain: video; Evidence: "we use the UCF101 video classification benchmark (Soomro et al., 2012)." (Section 5.3. 3D Classification)
- Task name: Multi-resolution ImageNet classification (resolution generalization); Task type: Classification; Dataset(s): ImageNet validation set; Domain: natural images; Evidence: "In this section we compare the ability of methods to generalize to image resolutions not seen during training." and "We evaluate the accuracy on the ImageNet validation set with varying inference resolutions." (Section 5.6. Multi-resolution Classification)

## 4. Domain and Modality Scope

- Single domain? No; evaluation spans "2D and 3D vision, spatial reasoning, and resolution generalization." (Section 5. Experiments)
- Multiple domains within the same modality? Yes; the modality is vision with multiple task types: "2D and 3D vision, spatial reasoning, and resolution generalization." (Section 5. Experiments)
- Multiple modalities? Not stated; the evaluation is described as "2D and 3D vision tasks" (Abstract).
- Domain generalization or cross-domain transfer? Resolution generalization is claimed: "showing that it generalizes well to higher input resolutions" (Abstract); cross-domain transfer is not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| 2D image classification (CIFAR-100, ImageNet-1k) | No (trained from scratch per task) | No | Not specified | "All models use ViT-based architectures trained from scratch with standard data augmentations (RandAugment)." (Section 5.1) |
| Synthetic spatial reasoning | No (no pre-trained weights stated) | No | Not specified | "We avoid using pre-trained weights in order to help reproducibility and comparability of the results between methods." (Section 5) |
| 3D video classification (UCF101) | No (trained from scratch per task) | No | Not specified | "All models use a ViT-style backbone with 3D patch tokenization, trained from scratch with no hyperparameter tuning" (Section 5.3) |
| Multi-resolution ImageNet classification | Not specified across tasks; within-task pretrain+fine-tune recipe | Yes (second recipe) | Not specified | "The second adds an additional fine-tuning step at size  $256 \times 256$  for 30 epochs." (Section 5.6) |

## 6. Input and Representation Constraints

- 2D patch sizes and resizing: "We use a patch size of  $4\times4$  on the original  $32\times32$  image for CIFAR-100 and a patch size of  $16\times16$  on the randomly cropped and resized  $224\times224$  image." (Appendix B.2. 2D Image Classification)
- 3D patch sizes and resizing: "a patch size of  $2 \times 16 \times 16$  on the randomly cropped and resized  $8 \times 224 \times 224$  video/image." (Appendix B.3. 3D Video Classifications)
- Synthetic task fixed grid and resolution: "The task presents a  $108 \times 108$  pixel image containing a  $9 \times 9$  grid (81 cells)." (Section 5.2)
- Synthetic task resolution variation: "We evaluate the models across three different input resolutions ( $108 \times 108, 168 \times 168, \text{ and } 276 \times 276 \text{ pixels}$ )" (Section 5.2)
- Multi-resolution training/inference sizes: "training the models on images of size  $224 \times 224$  for 200 epochs." and "The second adds an additional fine-tuning step at size  $256 \times 256$  for 30 epochs." and "We scale the input images to resolutions of  $196 \times 196$ ,  $256 \times 256$ ,  $320 \times 320$ ,  $384 \times 384$ , and  $448 \times 448$  pixels per dimension" (Section 5.6)
- Fixed dimensionality assumption: "Recall that we assume that positions are n-dimensional vectors" (Section 4) and "N denotes the number of input dimensions, i.e. N=3 for the 3D image." (Figure 1)
- Fixed number of tokens: Not specified.
- Padding requirements: Not specified beyond "randomly cropped and resized" inputs (Appendix B.2/B.3).

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Sequence length fixed or variable: Not specified; evaluation varies input resolution ("We scale the input images to resolutions of  $196 \times 196$ ,  $256 \times 256$ ,  $320 \times 320$ ,  $384 \times 384$ , and  $448 \times 448$  pixels per dimension") (Section 5.6).
- Attention type: standard softmax attention is used in the attention computation: "Attention  $\leftarrow$  softmax  $\left(\frac{Q_{\text{rot}} K_{\text{rot}}^T}{\sqrt{\dim(K)}}\right) V$" (Algorithm 1).
- Computational cost management: no explicit windowing/pooling described; they note that "runtime is dominated by the quadratic attention component" (Section D.1. FLOPS Comparison of methods).

## 8. Positional Encoding (Critical Section)

- Mechanism: LieRE is rotation-based and learnable: "Lie Relative Encodings (LieRE) introduce a principled generalization of RoPE" and "LieRE learns dense skew-symmetric matrices (Lie algebra elements), which are then differentiable mapped to form high-dimensional rotation matrices (Lie group elements)." (Abstract)
- Relative/absolute scope: "This results in richer, learnable, and continuous, encodings of both relative and absolute positional information." (Abstract)
- Where applied: "LieRE's final step is to modify token i's query and keys as  $Q_i' = R(p_i)Q_i$  and  $K_i' = R(p_i)K_i$ ." (Section 4) and "By default, the skew symmetric bases are learned separately for every layer and attention head except in the experimental section focused on sharing parameters across heads and layers." (Method)
- Fixed/modified/ablated: "We evaluate two versions of LieRE, distinguished by the basis matrix block-diagonal sizes of 64 and 8" (Section 5) and "We compare LieRE to absolute positional encodings, RoPE-Mixed (Heo et al., 2024), and VisionLLaMA (Chu et al., 2024)." (Section 5.1)

## 9. Positional Encoding as a Variable

- Core research variable: Yes; "To assess the impact of LieRE and other positional encodings on ViT performance, we evaluate several encoding schemes across diverse tasks" (Section 5).
- Multiple positional encodings compared: Yes; "We compare LieRE to absolute positional encodings, RoPE-Mixed (Heo et al., 2024), and VisionLLaMA (Chu et al., 2024)." (Section 5.1)
- PE choice "not critical" or secondary: Not stated.

## 10. Evidence of Constraint Masking

- Model sizes: "All models use 85.2M parameters for 2D tasks and 88.7M parameters for 3D task" (Table 1) and "synthetic task on base model (85M)" (Table 3).
- Model-scale sweeps: "We also investigate performance across model sizes (ViT-Tiny, Base, Large)." (Section 5.1)
- Dataset sizes: "We train the models on 800,000 examples and observe that they generally converge after the first 400,000 examples." (Section 5.2) and "training on only 20–90% of the CIFAR-100 dataset." (Section 5.1 / Figure 4).
- Scaling attribution: Not explicitly attributed to scaling model size or data; they state, "Our goal is to isolate the contribution of positional encodings by using consistent architectures, training procedures, and hyperparameters across all methods." (Section 5)
- Architectural capacity vs scale: "We control capacity via imposing a block-diagonal structure on the basis matrices. Smaller blocks (e.g.,  $2 \times 2$ ) replicate RoPE-Mixed, while larger blocks increase expressivity." (Section 5.4)

## 11. Architectural Workarounds

- 3D patch tokenization for video input: "All models use a ViT-style backbone with 3D patch tokenization" (Section 5.3).
- CLS pooling as aggregation: "We used CLS pooling in our implementation to facilitate comparability with existing literature in the field." (Appendix B.1)
- Block-diagonal basis to control PE capacity: "We control capacity via imposing a block-diagonal structure on the basis matrices. Smaller blocks (e.g.,  $2 \times 2$ ) replicate RoPE-Mixed, while larger blocks increase expressivity." (Section 5.4)
- Parameter sharing across heads/layers (capacity/efficiency study): "By default, the skew symmetric bases are learned separately for every layer and attention head except in the experimental section focused on sharing parameters across heads and layers." (Method)
- Fixed grid assumption in synthetic task: "The task presents a  $108 \times 108$  pixel image containing a  $9 \times 9$  grid (81 cells)." (Section 5.2)

## 12. Explicit Limitations and Non-Claims

- "While LieRE shows promising results for 2D and 3D inputs, several limitations are worth noting." (Section 7. Limitations)
- "For 1D input, LieRE reduces to RoPE with learnable phases (proof in appendix A)." (Section 7. Limitations)
- "However, this may limit its applicability to other architectures—such as convolutional neural networks—that do not rely on the attention mechanism." (Section 7. Limitations)
- "The current formulation encodes vector positions in  $\mathbb{R}^d$ . While sufficient for many applications, it may not directly apply to tasks that require pose encoding in SE(3) (e.g., robotics)." (Section 7. Limitations)
- "Lastly, in its current implementation, LieRE relies on the accuracy and numerical stability of the matrix exponential in PyTorch. Future work may explore more efficient and robust implementations or approximations of this operation." (Section 7. Limitations)
- Explicit non-claims about unrestrained multi-task learning or open-world learning: Not stated.

### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: Vision-only evaluation across 2D images, 3D videos, and synthetic grids.
- Task structure: Supervised classification tasks (CIFAR-100, ImageNet-1k, UCF101) plus a synthetic spatial reasoning classification task and resolution generalization evaluations.
- Representation rigidity: Fixed patch sizes and explicit input resolutions (e.g., 32x32, 224x224, 8x224x224; fixed 9x9 grid for the synthetic task).
- Model sharing vs specialization: Models trained from scratch per task with no pre-trained weights, with a separate fine-tuning step only in the multi-resolution recipe.
- Role of positional encoding: Central variable with LieRE variants (block sizes) and comparisons against absolute and RoPE-based encodings.

### 14. Final Classification

Classification: **Multi-task, multi-domain (constrained)**. The paper evaluates multiple tasks across distinct visual domains, including "2D and 3D vision, spatial reasoning, and resolution generalization" (Section 5) with datasets like "CIFAR-100 and ImageNet-1k" (Section 5.1) and "the UCF101 video classification benchmark" (Section 5.3). Cross-domain transfer beyond vision is not claimed.
