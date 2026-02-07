# LOOPE: Learnable Optimal Patch Order in Positional Embeddings for Vision Transformers (Not specified in the paper)
Source: LOOPE- Learnable Optimal Patch Order in Vision Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | images | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | class label (inferred) | 0D (inferred) | Fixed (inferred) |
| Image classification (three-cell positional relations) | synthetic RGB images (three-cell grid) | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | 6-class label (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates LOOPE on supervised image classification, covering standard datasets (Oxford-IIIT, CIFAR-100) and a synthetic three-cell positional-relation benchmark. Inputs are 2D RGB images at fixed resolutions/grids (e.g., 224x224 and 14x14), and outputs are single class labels. Attention dynamics and state dynamics are not explicitly specified in the OCR, so those fields remain unspecified.

## Evidence
### Task: Image classification
- "Empirical results show that our PE significantly improves classification accuracy across various ViT architectures." (Abstract)
- "We evaluate the effectiveness of different positional encodings on Vision Transformer architectures using the Oxford-IIIT and CIFAR-100 datasets." (Section 4.2. Comparison with 1-D Positional Embeddings)
- "In case of CrossViT, we used  $240\times240$  images with mixed patch sizes ( $12\times12$ ,  $16\times16$ )." (Section 4.1. Experimental Setup)
- Inference: In Dimension and In Dynamics inferred from fixed image sizes ("240x240 images") and resolution-specific experiments; Output/Out Dimension/Out Dynamics inferred from "classification accuracy" implying a single label per image. (Abstract; Section 4.1)

### Task: Image classification (three-cell positional relations)
- "we construct a synthetic dataset of  $224 \times 224$ RGB images" (Section 3.2. Three Cell Experiment)
- "each synthetic image  $I_s$  is partitioned into a  $14 \times 14$  grid" (Section 3.2. Three Cell Experiment)
- "a simple 6-class image classification task is enough." (Section 3.2. Three Cell Experiment)
- Inference: In Dimension and In Dynamics inferred from fixed 224x224 images and a 14x14 grid; Output/Out Dimension/Out Dynamics inferred from the 6-class classification wording (single label output). (Section 3.2)
