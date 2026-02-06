# ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases (2021)
Source: ConViT- Improving Vision Transformers with Soft Convolutional Inductive Biases.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image classification | images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | class labels | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates ConViT on image classification, reporting results on ImageNet-1k (including subsampled variants) and CIFAR100. The inputs are fixed-size 224x224 images represented as a 2D patch grid, producing a single class label per image, so the justified dimensions are 2D in and 0D out with fixed dynamics. Based on the described ViT/ConViT architecture with self-attention over a fixed patch sequence and no persistent memory, attention is static and state is direct (inferred).

## Evidence
### Task: image classification
- "Vision Transformers (ViTs) rely on more flexible self-attention layers, and have recently outperformed CNNs for image classification." (Section: Abstract)
- "The ViT slices input images of size 224 into  $16 \times 16$  non-overlapping patches" (Section: Architectural details)
- "predict the class of the input." (Section: Architectural details)
- "performing SA across embeddings of patches of pixels." (Section: Introduction)
- Inference: In Dimension 2D (x, y) and Fixed input dynamics inferred from fixed 224-sized images and patch grid; Out Dimension 0D and Fixed output dynamics inferred from class prediction; Static attention and Direct state inferred from self-attention over the fixed patch sequence without any described external memory (supported by the quotes above).
