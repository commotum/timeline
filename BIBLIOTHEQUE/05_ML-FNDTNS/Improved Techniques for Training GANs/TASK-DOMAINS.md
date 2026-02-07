# Improved Techniques for Training GANs (Not specified in the paper.)
Source: Improved Techniques for Training GANs.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image generation | noise vector z | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | images (samples from data distribution) | 2D (x, y) (inferred) | Fixed (inferred) |
| classification (semi-supervised) | images (data point x) | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | class logits/probabilities (K or K+1 classes) | 1D (t) (inferred) | Fixed (inferred) |

## Summary
The paper applies GANs to image generation and semi-supervised image classification. Inputs include noise vectors and images, producing images and class logits/probabilities, implying 1D and 2D domains with fixed sizes (inferred from the fixed-resolution datasets and K-dimensional logits). Attention and state dynamics are not specified in the paper.

## Evidence
### Task: image generation
- "the generation of images that humans find visually realistic." (Abstract)
- "transforming vectors of noise  $\\boldsymbol{z}$" (Section 1 Introduction)
- "dataset of  $32 \\times 32$  natural images." (Section 6.2 CIFAR-10)
- "$128 \\times 128$  images from the ILSVRC2012 dataset" (Section 6.4 ImageNet)
- Inference: In Dimension/In Dynamics are 1D and Fixed because the generator input is described as noise vectors; Out Dimension/Out Dynamics are 2D and Fixed because images are described at fixed resolutions (32x32, 128x128).

### Task: classification (semi-supervised)
- "semi-supervised classification on MNIST, CIFAR-10 and SVHN." (Abstract)
- "takes in x as input and outputs a K-dimensional vector of logits" (Section 5 Semi-supervised learning)
- "labeling them with a new \"generated\" class y=K+1" (Section 5 Semi-supervised learning)
- "dataset of  $32 \\times 32$  natural images." (Section 6.2 CIFAR-10)
- Inference: In Dimension/In Dynamics are 2D and Fixed because the inputs are fixed-resolution images; Out Dimension/Out Dynamics are 1D and Fixed because the classifier outputs a K-dimensional (or K+1) logit vector.
