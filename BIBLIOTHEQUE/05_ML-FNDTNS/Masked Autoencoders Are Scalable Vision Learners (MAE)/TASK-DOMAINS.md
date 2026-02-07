# Masked Autoencoders Are Scalable Vision Learners (Not specified in the paper.)
Source: Masked Autoencoders Are Scalable Vision Learners (MAE).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked image reconstruction | images with masked patches | 2D (x, y) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | reconstructed image pixels | 2D (x, y) | Fixed (inferred) |
| image classification | images | 2D (x, y) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | class label | 0D | Not specified in the paper. |
| object detection | images | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | bounding boxes | 2D (x, y) | Not specified in the paper. |
| instance segmentation | images | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | instance masks | 2D (x, y) | Not specified in the paper. |
| semantic segmentation | images | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | per-pixel semantic labels | 2D (x, y) | Not specified in the paper. |

## Summary
The paper centers on masked image reconstruction for self-supervised pretraining and evaluates the learned representations on downstream image classification, object detection, instance segmentation, and semantic segmentation. Inputs are consistently 2D images, with outputs that are reconstructed pixels for the pretext task, 0D class labels for classification, and 2D spatial outputs for detection/segmentation. Fixed-size imagery is described for ImageNet evaluation, while most downstream input/output dynamics are not explicitly specified; attention and state properties are inferred from the encoder–decoder architecture.

## Evidence
### Task: masked image reconstruction
- "we mask random patches of the input image and reconstruct the missing pixels." (Abstract)
- "decoder that reconstructs the original image in pixels." (Figure 1 caption)
- Inference: In/Out Dynamics set to Fixed (inferred) based on "a single  $224 \times 224$  crop" (4. ImageNet Experiments) and "39 out of 196 patches" (Figure 2 caption); Attention Dynamic set to Static (inferred) because "the full set of encoded patches and mask tokens is processed by a small decoder" (Figure 1 caption); State Dynamic set to Constructed (inferred) because "encoder that maps the observed signal to a latent representation" (3. Approach).

### Task: image classification
- "the encoder is applied to uncorrupted images (full sets of patches) for recognition tasks." (Figure 1 caption)
- "Classification tasks. Table 6 studies transfer learning on the iNaturalists [56] and Places [71] tasks" (5. Transfer Learning Experiments)
- Inference: In Dynamics set to Fixed (inferred) based on "a single  $224 \times 224$  crop" (4. ImageNet Experiments) and "image size of 224, except for ViT-H with an extra result on 448" (Table 3 caption); Attention Dynamic set to Static (inferred) because "the full set of encoded patches and mask tokens is processed by a small decoder" (Figure 1 caption); State Dynamic set to Constructed (inferred) because "encoder that maps the observed signal to a latent representation" (3. Approach).

### Task: object detection
- "We also evaluate transfer learning on object detection, instance segmentation, and semantic segmentation." (1. Introduction)
- "We report box AP for object detection" (5. Transfer Learning Experiments)
- Inference: Attention Dynamic set to Static (inferred) because "the full set of encoded patches and mask tokens is processed by a small decoder" (Figure 1 caption); State Dynamic set to Constructed (inferred) because "encoder that maps the observed signal to a latent representation" (3. Approach).

### Task: instance segmentation
- "We also evaluate transfer learning on object detection, instance segmentation, and semantic segmentation." (1. Introduction)
- "We report box AP for object detection and mask AP for instance segmentation." (5. Transfer Learning Experiments)
- Inference: Attention Dynamic set to Static (inferred) because "the full set of encoded patches and mask tokens is processed by a small decoder" (Figure 1 caption); State Dynamic set to Constructed (inferred) because "encoder that maps the observed signal to a latent representation" (3. Approach).

### Task: semantic segmentation
- "We also evaluate transfer learning on object detection, instance segmentation, and semantic segmentation." (1. Introduction)
- "Semantic segmentation. We experiment on ADE20K [72] using UperNet [63]" (5. Transfer Learning Experiments)
- Inference: Attention Dynamic set to Static (inferred) because "the full set of encoded patches and mask tokens is processed by a small decoder" (Figure 1 caption); State Dynamic set to Constructed (inferred) because "encoder that maps the observed signal to a latent representation" (3. Approach).
