# ImageNet Classification with Deep Convolutional Neural Networks (Not specified in the paper.)
Source: ImageNet Classification with Deep Convolutional Neural Networks (AlexNet).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | images (RGB images) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | class labels (1000 classes) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper describes supervised image classification, mapping RGB images to a fixed set of 1000 class labels. Inputs are fixed-size 2D images, and outputs are fixed-size label distributions; no runtime input selection is described, so attention is static and the mapping is direct. Overall, the task coverage is limited to single-image classification with fixed spatial dimensions and fixed label space.

## Evidence
### Task: classification
- "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes." (Abstract)
- "ImageNet consists of variable-resolution images, while our system requires a constant input dimensionality." (Section 2 The Dataset)
- "we down-sampled the images to a fixed resolution of  $256 \times 256$ ." (Section 2 The Dataset)
- "The output of the last fully-connected layer is fed to a 1000-way softmax which produces a distribution over the 1000 class labels." (Section 3.5 Overall Architecture)
- Inference: In Dimension is 2D (x, y) and In Dynamics are Fixed because the model uses fixed-resolution image inputs; Attention is Static and State is Direct because the paper describes a feedforward classifier with no runtime selection or persistent state; Out Dimension is 0D and Out Dynamics are Fixed because outputs are 1000-way class labels. (Supported by the fixed-resolution input and 1000-way softmax descriptions in Sections 2 and 3.5.)
