## 1. Basic Metadata

Title: LOOPE: Learnable Optimal Patch Order in Positional Embeddings for Vision Transformers
Authors: Md Abtahi Majeed Chowdhury; Md Rifat Ur Rahman; Akil Ahmad Taki
Year: Year not specified.
Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

The paper proposes LOOPE, a learnable patch-ordering method for positional embeddings that optimizes spatial representation for a fixed set of frequencies and evaluates positional information retention with the Three-Cell Experiment.

## 3. Tasks Evaluated

### Task 1
- Task name: Image classification on Oxford-IIIT and CIFAR-100
- Task type: Classification
- Dataset(s) used: Oxford-IIIT; CIFAR-100
- Domain: Not explicitly stated (Vision Transformer architectures are named, but the dataset domains are not described)
- Evidence (quotes):
  - "We evaluate the effectiveness of different positional encodings on Vision Transformer architectures using the Oxford-IIIT and CIFAR-100 datasets." (Section 4.2. Comparison with 1-D Positional Embeddings)
  - "Empirical results show that our PE significantly improves classification accuracy across various ViT architectures." (Abstract)

### Task 2
- Task name: Three-Cell Experiment (synthetic 6-class image classification)
- Task type: Classification
- Dataset(s) used: synthetic dataset / Three cell dataset
- Domain: Synthetic RGB images / synthetic grid
- Evidence (quotes):
  - "To mitigate this confounding factor, we construct a synthetic dataset of  $224 \times 224$ RGB images, ensuring that no two neighboring  $16 \times 16$ patches share common color information." (Section 3.2. Three Cell Experiment)
  - "Formally, each synthetic image  $I_s$  is partitioned into a  $14 \times 14$  grid, where three independent, non-overlapping cells are randomly assigned R, G, B." (Section 3.2. Three Cell Experiment)
  - "To evaluate all those case, a simple 6-class image classification task is enough." (Section 3.2. Three Cell Experiment)
  - "We evaluate positional encoders on four metrics: Distance, Orientation, Area, and Vector Sum." (Section 4.3. Analysis of Positional Encoding Performance in the Three-Cell Experiment)

## 4. Domain and Modality Scope

- Is evaluation performed on a single domain? No; it includes multiple datasets and a synthetic dataset: "Oxford-IIIT and CIFAR-100 datasets" and "a synthetic dataset of  $224 \times 224$ RGB images." (Section 4.2; Section 3.2)
- Multiple domains within the same modality? Yes, multiple image datasets plus a synthetic RGB image dataset are used (Section 4.2; Section 3.2).
- Multiple modalities? Not stated; the only explicit modality mentioned is "RGB images." (Section 3.2)
- Does the paper claim domain generalization or cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Oxford-IIIT classification | Not specified (ImageNet-1K pretrained weights are mentioned) | Not specified | Not specified | "Batch sizes were 96 for Oxford-IIIT, and 64 for CIFAR-100 and our novel Three cell dataset ." (Section 4.1. Experimental Setup); "All models were used with ImageNet-1K pretrained weights for a baseline comparison with the other experiments." (Section 4.1. Experimental Setup) |
| CIFAR-100 classification | Not specified (ImageNet-1K pretrained weights are mentioned) | Not specified | Not specified | "Batch sizes were 96 for Oxford-IIIT, and 64 for CIFAR-100 and our novel Three cell dataset ." (Section 4.1. Experimental Setup); "All models were used with ImageNet-1K pretrained weights for a baseline comparison with the other experiments." (Section 4.1. Experimental Setup) |
| Three-Cell synthetic classification | Not specified (ImageNet-1K pretrained weights are mentioned) | Not specified | Not specified | "Batch sizes were 96 for Oxford-IIIT, and 64 for CIFAR-100 and our novel Three cell dataset ." (Section 4.1. Experimental Setup); "All models were used with ImageNet-1K pretrained weights for a baseline comparison with the other experiments." (Section 4.1. Experimental Setup) |

## 6. Input and Representation Constraints

- Fixed resolution and patch size for the Three-Cell dataset: "we construct a synthetic dataset of  $224 \times 224$ RGB images, ensuring that no two neighboring  $16 \times 16$ patches share common color information." (Section 3.2. Three Cell Experiment)
- Fixed grid size in the Three-Cell dataset: "each synthetic image  $I_s$  is partitioned into a  $14 \times 14$  grid." (Section 3.2. Three Cell Experiment)
- 2D grid to 1D sequence assumption: "Mapping an N-dimensional grid to a 1D sequence while preserving specific properties is a key challenge in computational science." (Section 3. Methodology)
- Fixed 2D grid constraint for Hilbert order: "The Hilbert curve maps a  $2^n \times 2^n$  grid to a 1D sequence while preserving spatial locality but cannot handle arbitrary rectangular grids." (Section 3.1. Proposed Method)
- Mixed patch sizes for one architecture: "In case of CrossViT, we used  $240\times240$  images with mixed patch sizes ( $12\times12$ ,  $16\times16$ )." (Section 4.1. Experimental Setup)
- Multiple input resolutions across experiments: "At 224x224 gains range from +2.8% to +3.5%." and "At 384x384, performance gains improves significantly, particularly in DeiT-Base(+3.9%), demonstrating LOOPE's greater improvement for bigger resolution." (Section 4.2. Comparison with 1-D Positional Embeddings)
- Padding/resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; the only explicit token grid size is "a  $14 \times 14$  grid" for the synthetic dataset. (Section 3.2. Three Cell Experiment)
- Fixed or variable sequence length: Fixed within the Three-Cell dataset ("$14 \times 14$  grid"), but variable across experiments due to different resolutions ("At 224x224" and "At 384x384"). (Section 3.2; Section 4.2)
- Attention type: Not specified; only self-attention is mentioned in general terms: "permutation-invariant nature of selfattention." (Abstract)
- Mechanisms to manage computational cost: Not specified.

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: LOOPE learns a patch order and applies sinusoidal sin/cos embeddings: "we propose LOOPE, a learnable patch-ordering method that optimizes spatial representation for a given set of frequencies" (Abstract) and "$$\mathbf{E}(\mathbf{X}) = \mathbf{E}(\mathbf{X_G} + \mathbf{X_C}) = \mathbf{sin}(\mathbf{XW^T}) | \mathbf{cos}(\mathbf{XW^T})$$" (Section 3.1. Proposed Method).
- Where it is applied: Not specified.
- Fixed across experiments vs modified: The PE choice is varied across experiments: "each trained with five positional encoding methods: Zero PE, Learnable PE, Sinusoidal PE, Hilbert PE, and our proposed Learnable Hilbert PE." (Section 4.2. Comparison with 1-D Positional Embeddings)

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption? Core research variable: "Positional embeddings (PE) play a crucial role in Vision Transformers (ViTs)" (Abstract) and "We evaluate the effectiveness of different positional encodings" (Section 4.2. Comparison with 1-D Positional Embeddings).
- Are multiple positional encodings compared? Yes: "each trained with five positional encoding methods: Zero PE, Learnable PE, Sinusoidal PE, Hilbert PE, and our proposed Learnable Hilbert PE." (Section 4.2. Comparison with 1-D Positional Embeddings)
- Does the paper claim PE choice is "not critical" or secondary? Not claimed.

## 10. Evidence of Constraint Masking

- Model sizes: Not specified; model variants are listed without size details: "The tested models include ViT-Base, DeiT-Base, DeiT-Small, CaiT, and Cross-ViT." (Section 4.2. Comparison with 1-D Positional Embeddings)
- Dataset sizes: Not specified.
- Performance gains attribution (architecture): "LOOPE further enhances accuracy, because it integrates spatial locality with learnable flexibility, leading to a more effective representation of positional dependencies." (Section 4.2. Comparison with 1-D Positional Embeddings)
- Scaling input resolution: "Our method shows higher improvement in accuracy with bigger resolution. At 224x224 gains range from +2.8% to +3.5%. ... At 384x384, performance gains improves significantly, particularly in DeiT-Base(+3.9%), demonstrating LOOPE's greater improvement for bigger resolution." (Section 4.2. Comparison with 1-D Positional Embeddings)
- Training tricks: "All models were used with ImageNet-1K pretrained weights for a baseline comparison with the other experiments." (Section 4.1. Experimental Setup)

## 11. Architectural Workarounds

- 2D grid flattening as an architectural step: "Mapping an N-dimensional grid to a 1D sequence while preserving specific properties is a key challenge in computational science." (Section 3. Methodology)
- Learnable patch ordering with static and contextual components: "We are proposing a Learnable patch ordering method which generates stable yet dynamic order,  $\mathbf{X}$ , combining with  $\mathbf{X}_{\mathbf{G}}$  and  $\mathbf{X}_{\mathbf{C}}$ ,  $\mathbf{X} = \mathbf{X}_{\mathbf{G}} + \mathbf{X}_{\mathbf{C}}$  where,  $\mathbf{X}_{\mathbf{G}}$  is fractal curve order and  $\mathbf{X}_{\mathbf{C}}$  is context bias." (Section 3. Methodology)
- Generalized Hilbert/Gilbert order for arbitrary shapes: "To generate patch order for arbitrary image shape, we used generalized Hilbert order, also known as Gilbert Order[33], which generates SFC for arbitrary 2D dimensions by recursively dividing the grid while maintaining locality." (Section 3.1. Proposed Method)
- Context bias to locally adjust order: "To leverage contextual information for patch ordering, we propose a controlled mechanism that locally manipulates the static Gilbert order,  $\mathbf{X}_{\mathbf{G}}$ ." (Section 3.1. Proposed Method)
- Mixed patch sizes in CrossViT experiments: "we used  $240\times240$  images with mixed patch sizes ( $12\times12$ ,  $16\times16$ )." (Section 4.1. Experimental Setup)

## 12. Explicit Limitations and Non-Claims

- "While our proposed LOOPE framework does not claim to deliver state-of-the-art results across all dimensions, it establishes a solid foundation for future research to further investigate and refine positional embeddings in vision models." (Section 5. Conclusion)

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Multiple image datasets plus a synthetic RGB image dataset; no multi-modal evaluation.
> - Task structure: Supervised image classification, including a synthetic 6-class positional-relation task.
> - Representation rigidity: Fixed patch grids within datasets (e.g., $14 \times 14$ grid) and 2D-to-1D sequence mapping; input resolution varies across experiments.
> - Model sharing vs specialization: Separate training setups per dataset with ImageNet-1K pretrained weights; no joint multi-task training described.
> - Role of positional encoding: Central experimental variable with multiple PE variants compared and ablated.

### 14. Final Classification

**Multi-task, multi-domain (constrained).** The paper evaluates multiple classification tasks across different datasets, including "Oxford-IIIT and CIFAR-100 datasets" and a "synthetic dataset of  $224 \times 224$ RGB images" (Section 4.2; Section 3.2). All evaluations stay within the vision modality and do not claim cross-domain transfer, indicating a constrained multi-domain setup rather than unrestrained multi-task learning.
