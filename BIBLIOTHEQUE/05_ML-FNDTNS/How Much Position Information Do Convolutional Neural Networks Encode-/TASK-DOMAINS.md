# How Much Position Information Do Convolutional Neural Networks Encode? (Not specified in the paper)
Source: How Much Position Information Do Convolutional Neural Networks Encode-.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| position map prediction | images | 2D (x, y) | Fixed | Static (inferred) | Constructed (inferred) | gradient-like position map | 2D (x, y) | Fixed |
| salient object detection | images (inferred) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | saliency map (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |
| semantic segmentation | images (inferred) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | segmentation map (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |

## Summary
The paper’s primary task is predicting absolute position maps from images, using a fixed image size in experiments. It also evaluates position-dependence via salient object detection and semantic segmentation, which implies 2D image-grid domains for those tasks. Attention/state dynamics are inferred as static/constructed for PosENet, while dynamics for the downstream tasks are not specified.

## Evidence
### Task: position map prediction
- "Given an input image  $\mathcal{I}_m \in \mathbb{R}^{h \times w \times 3}$ , our goal is to predict a gradient-like position information mask" (Section 2, Problem Formulation)
- "We resize each image to a fixed size of  $224 \times 224$  during training and inference." (Section 3.2, Implementation Details)
- Inference: Attention Dynamic = Static (inferred) and State Dynamic = Constructed (inferred) because the model is a "feed-forward convolutional encoder network" that extracts multi-scale features. (Section 2.1, Position Encoding Network)

### Task: salient object detection
- "Saliency Detection: We further validate our findings in the position-dependent tasks (semantic segmentation and salient object detection (SOD))." (Section 4.2, Zero-Padding Driven Position Information)
- "the regions determined to be most salient (Jia & Bruce, 2018) tend to be near the center of an image." (Section 1, Introduction)
- "critical for detecting salient regions." (Section 4.2, Zero-Padding Driven Position Information)
- Inference: Input/Output are image grids and a saliency map (inferred), and dimensions are 2D (x, y) (inferred), based on the task being salient object detection of regions in images. (Section 4.2, Zero-Padding Driven Position Information)

### Task: semantic segmentation
- "Semantic Segmentation: We also validate the impact of zero-padding on the semantic segmentation task." (Section 4.2, Zero-Padding Driven Position Information)
- "We train the VGG16 network with and without zero padding on the training set of PASCAL VOC 2012 dataset" (Section 4.2, Zero-Padding Driven Position Information)
- Inference: Input/Output are image grids and a segmentation map (inferred), and dimensions are 2D (x, y) (inferred), based on the semantic segmentation task on images. (Section 4.2, Zero-Padding Driven Position Information)
