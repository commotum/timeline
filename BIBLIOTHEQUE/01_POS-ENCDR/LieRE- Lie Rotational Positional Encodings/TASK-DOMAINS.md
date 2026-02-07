# LieRE: Lie Rotational Positional Encodings (2025)
Source: LieRE- Lie Rotational Positional Encodings.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2D image classification (CIFAR-100, ImageNet-1k) | Images | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Class label (inferred) | 0D (inferred) | Fixed (inferred) |
| Synthetic spatial reasoning classification (arrow direction) | Synthetic grid images | 2D (x, y) | Capped (inferred) | Static (inferred) | Direct (inferred) | Arrow direction label (inferred) | 0D (inferred) | Fixed (inferred) |
| 3D video classification (UCF101) | Videos | 3D (x, y, t) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Class label (inferred) | 0D (inferred) | Fixed (inferred) |
| Image classification (multi-resolution generalization) | Images | 2D (x, y) | Capped (inferred) | Static (inferred) | Direct (inferred) | Class label (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
LieRE is evaluated only on vision classification tasks: standard 2D image classification (CIFAR-100, ImageNet-1k), a synthetic spatial reasoning image task, and 3D video classification (UCF101), plus an ImageNet resolution generalization evaluation. Inputs are 2D images or 3D videos, with explicit resolution ranges indicating capped input dynamics in the synthetic and multi-resolution settings, while other input dynamics are not specified. Outputs are categorical labels (inferred), and attention/state dynamics are inferred as static/direct based on the use of standard ViT-style attention.

## Evidence
### Task: 2D image classification (CIFAR-100, ImageNet-1k)
- "We begin with CIFAR-100 and ImageNet-1k benchmarks to evaluate LieRE in 2D vision tasks." (Section 5.1. 2D Image Classification)
- "Experiments on 2D image classification (CIFAR-100, ImageNet-1k) and 3D video classification (UCF101)" (Section 6. Conclusion)
- Inference: Output treated as class labels (0D, Fixed) and attention/state marked Static/Direct based on ViT-based models and standard attention ("All models use ViT-based architectures trained from scratch with standard data augmentations (RandAugment)." (Section 5.1) and "Attention  $\leftarrow$  softmax  $\left(\frac{Q_{\text{rot}} K_{\text{rot}}^T}{\sqrt{\dim(K)}}\right) V$" (Algorithm 1)).

### Task: Synthetic spatial reasoning classification (arrow direction)
- "we designed a synthetic image classification task (Shah et al., 2024)." (Section 5.2. Synthetic Spatial Reasoning Task)
- "The task presents a  $108 \times 108$  pixel image containing a  $9 \times 9$  grid (81 cells)." (Section 5.2. Synthetic Spatial Reasoning Task)
- "The objective is to identify the direction of this specific arrow." (Section 5.2. Synthetic Spatial Reasoning Task)
- "We evaluate the models across three different input resolutions ( $108 \times 108, 168 \times 168, \text{ and } 276 \times 276 \text{ pixels}$ )." (Section 5.2. Synthetic Spatial Reasoning Task)
- Inference: Capped input dynamics inferred from the multiple stated input resolutions; output treated as an arrow-direction label (0D, Fixed) and attention/state marked Static/Direct based on standard attention ("Attention  $\leftarrow$  softmax  $\left(\frac{Q_{\text{rot}} K_{\text{rot}}^T}{\sqrt{\dim(K)}}\right) V$" (Algorithm 1)).

### Task: 3D video classification (UCF101)
- "To assess LieRE's performance on 3D data, we use the UCF101 video classification benchmark (Soomro et al., 2012)." (Section 5.3. 3D Classification)
- "Experiments on 2D image classification (CIFAR-100, ImageNet-1k) and 3D video classification (UCF101)" (Section 6. Conclusion)
- Inference: Output treated as class labels (0D, Fixed) and attention/state marked Static/Direct based on standard attention ("Attention  $\leftarrow$  softmax  $\left(\frac{Q_{\text{rot}} K_{\text{rot}}^T}{\sqrt{\dim(K)}}\right) V$" (Algorithm 1)).

### Task: Image classification (multi-resolution generalization)
- "In this section we compare the ability of methods to generalize to image resolutions not seen during training." (Section 5.6. Multi-resolution Classification)
- "We evaluate the accuracy on the ImageNet validation set with varying inference resolutions." (Section 5.6. Multi-resolution Classification)
- "We scale the input images to resolutions of  $196 \times 196$ ,  $256 \times 256$ ,  $320 \times 320$ ,  $384 \times 384$ , and  $448 \times 448$  pixels per dimension" (Section 5.6. Multi-resolution Classification)
- Inference: Capped input dynamics inferred from the explicit list of evaluated resolutions; output treated as class labels (0D, Fixed) and attention/state marked Static/Direct based on standard attention ("Attention  $\leftarrow$  softmax  $\left(\frac{Q_{\text{rot}} K_{\text{rot}}^T}{\sqrt{\dim(K)}}\right) V$" (Algorithm 1)).
