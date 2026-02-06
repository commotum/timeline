# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (Not specified in the paper.)
Source: Batch Normalization- Accelerating Deep Network Training by Reducing Internal Covariate Shift.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Digit classification (MNIST) | 28x28 binary image | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | digit class label | 0D (inferred) | Fixed (inferred) |
| Image classification (ImageNet, 1000 classes) | images | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | image class label (1000 classes) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates Batch Normalization on two image classification tasks: MNIST digit classification and ImageNet 1000-class classification. Inputs are fixed-size 2D images and outputs are fixed class labels, as supported by the 28x28 MNIST inputs and 224-resolution ImageNet crops. Attention and state dynamics are not specified in the paper.

## Evidence
### Task: Digit classification (MNIST)
- "predicting the digit class on the MNIST dataset" (Section 4.1)
- "with a 28x28 binary image as input" (Section 4.1)
- "10 activations (one per class)" (Section 4.1)
- Inference: In Dimension = 2D (x, y), In Dynamics = Fixed, Out Dimension = 0D, and Out Dynamics = Fixed, inferred from "28x28 binary image" and "10 activations (one per class)" (Section 4.1)

### Task: Image classification (ImageNet, 1000 classes)
- "trained on the ImageNet classification task" (Section 4.2)
- "softmax layer to predict the image class, out of 1000 possibilities" (Section 4.2)
- "BN-Inception single crop | 224" (Figure 4)
- Inference: In Dimension = 2D (x, y), In Dynamics = Fixed, Out Dimension = 0D, and Out Dynamics = Fixed, inferred from "image class" and "1000 possibilities" plus "BN-Inception single crop | 224" (Section 4.2, Figure 4)
