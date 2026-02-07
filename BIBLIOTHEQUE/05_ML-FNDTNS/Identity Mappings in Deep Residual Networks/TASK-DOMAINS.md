# Identity Mappings in Deep Residual Networks (Not specified in the paper)
Source: Identity Mappings in Deep Residual Networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image classification | images (inferred) | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | class labels (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates deep residual networks on image classification benchmarks including CIFAR-10/100 and ImageNet, reporting classification error. Inputs are 2D spatial crops with fixed sizes (e.g., 224×224 or 320×320), and outputs are single class labels (0D), inferred from the dataset descriptions. Attention and state dynamics are not specified in the paper.

## Evidence
### Task: image classification
- "Classification error on the CIFAR-10 test set using ResNet-110 [1]." (Table 1, Section 3.1 Experiments on Skip Connections)
- "Classification error (%) on the CIFAR-10/100 test set." (Table 3, Section 4.1 Experiments on Activation)
- "Next we report experimental results on the 1000-class ImageNet dataset [3]." (Section 5 Results)
- Inference: Input is 2D images with fixed-size crops and outputs are single class labels (0D) because the paper specifies "train crop 224×224", "test crop 320×320", and a "1000-class ImageNet dataset." (Table 5, Section 5 Results)
