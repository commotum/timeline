# DECOUPLING SEARCH AND LEARNING IN NEURAL NET TRAINING (Not specified in the paper.)
Source: Decoupling Search and Learning in Neural Net Training.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | Images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Class logits / class labels | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates a single downstream task domain: image classification on MNIST, CIFAR-10, and CIFAR-100. The task operates on images and predicts class outcomes, which supports a 2D (x, y) input space and 0D output decisions. The architecture and interfaces are fixed per model/dataset setup, and runtime attention is static. The method explicitly builds and uses layerwise representations as optimization targets, supporting a constructed state characterization.

## Evidence
### Task: Image classification
- "We apply our method to a standard convolutional network for CIFAR-10 classification." (Section 3.1)
- "To assess whether regressing to searched representations yields competitive generalization, we compare against standard stochastic gradient descent (SGD) training on MNIST, CIFAR-10, and CIFAR-100." (Section 4.3)
- "The final convolutional features are flattened and fed to a linear layer that outputs class logits." (Section 3.1)
- Inference: In Dimension, In Dynamics, Attention Dynamic, State Dynamic, Out Dimension, and Out Dynamics are inferred from the fixed image-CNN pipeline and explicit representation construction: "Data augmentation: Random horizontal flips and random crops from images padded by 2 pixels." (Section A.3), "The network consists of three convolutional blocks followed by a linear classification head." (Section 3.1), and "These cached representations become fixed regression targets—we never re-run search during training." (Section 4.2).
