# Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (Not specified in the paper.)
Source: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generation (image synthesis) | uniform noise distribution Z | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | generated images | 2D (x, y) (inferred) | Fixed (inferred) |
| classification (real vs generated discrimination) | images (real or generated) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | single sigmoid probability (real vs generated) | 0D (inferred) | Fixed (inferred) |
| classification (image classification via discriminator features) | images (CIFAR-10/SVHN) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | class labels via linear L2-SVM | 0D (inferred) | Fixed (inferred) |
| manipulation (semantic image manipulation) | Z vectors; generator feature maps (inferred) | 1D (t); 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | manipulated generated images | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper covers image-domain generation and classification, plus explicit semantic manipulation of generated samples. The generator maps fixed-size latent vectors to fixed-size 2D images, while discriminator-based pipelines map images to scalar decisions (real/fake or class labels). The supported dimensional range is 1D latent vectors, 2D image grids, and 0D label/probability outputs, with mostly Fixed dynamics. Attention is inferable as Static from fixed feed-forward CNN pipelines, and state is Direct for core adversarial mappings but Constructed when learned representations are explicitly reused/manipulated.

## Evidence
### Task: generation (image synthesis)
- "We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning." (Abstract)
- "Figure 1: DCGAN generator used for LSUN scene modeling. A 100 dimensional uniform distribution Z is projected to a small spatial extent convolutional representation with many feature maps. A series of four fractionally-strided convolutions (in some recent papers, these are wrongly called deconvolutions) then convert this high level representation into a  $64 \times 64$  pixel image. Notably, no fully connected or pooling layers are used." (Section 4, Figure 1 caption)
- Inference: `1D (t)`, `2D (x, y)`, `Fixed`, `Static`, and `Direct` are inferred from the explicit fixed-size latent vector to fixed-size image mapping in a single feed-forward generator path (Section 4, Figure 1 caption).

### Task: classification (real vs generated discrimination)
- "The first layer of the GAN, which takes a uniform noise distribution Z as input, could be called fully connected as it is just a matrix multiplication, but the result is reshaped into a 4-dimensional tensor and used as the start of the convolution stack." (Section 3 Approach and Model Architecture)
- "For the discriminator, the last convolution layer is flattened and then fed into a single sigmoid output." (Section 3 Approach and Model Architecture)
- Inference: The task intent `classification (real vs generated discrimination)` and labels `2D (x, y)`, `0D`, `Fixed`, `Static`, and `Direct` are inferred from the adversarial discriminator design with image input and a single sigmoid scalar output (Section 3).

### Task: classification (image classification via discriminator features)
- "We use the trained discriminators for image classification tasks, showing competitive performance with other unsupervised algorithms." (Section 1 Introduction, contributions list)
- "To evaluate the quality of the representations learned by DCGANs for supervised tasks, we train on Imagenet-1k and then use the discriminator's convolutional features from all layers, maxpooling each layers representation to produce a  $4 \times 4$  spatial grid. These features are then flattened and concatenated to form a 28672 dimensional vector and a regularized linear L2-SVM classifier is trained on top of them." (Section 5.1)
- "On the StreetView House Numbers dataset (SVHN)(Netzer et al., 2011), we use the features of the discriminator of a DCGAN for supervised purposes when labeled data is scarce." (Section 5.2)
- Inference: `2D (x, y)`, `0D`, `Fixed`, and `Static` are inferred from fixed-size image feature extraction and fixed-label supervised classification setup; `Constructed` is inferred because learned discriminator representations are explicitly extracted and reused as first-class features for downstream classifiers (Sections 5.1 and 5.2).

### Task: manipulation (semantic image manipulation)
- "• We show that the generators have interesting vector arithmetic properties allowing for easy manipulation of many semantic qualities of generated samples." (Section 1 Introduction, contributions list)
- "Using this simple model, all feature maps with weights greater than zero (200 in total) were dropped from all spatial locations. Then, random new samples were generated with and without the feature map removal." (Section 6.3.1)
- "We performed similar arithmetic on the Z vectors of sets of exemplar samples for visual concepts." (Section 6.3.2)
- Inference: `1D (t); 2D (x, y)`, `Fixed`, `Static`, and `Constructed` are inferred because manipulation is performed through learned latent vectors (`Z`) and learned spatial feature maps to control generated image content in fixed-shape generator pipelines (Sections 6.3.1 and 6.3.2).
