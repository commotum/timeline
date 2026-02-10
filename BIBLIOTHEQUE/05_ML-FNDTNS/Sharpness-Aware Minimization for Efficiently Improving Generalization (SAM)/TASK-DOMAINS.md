# SHARPNESS-AWARE MINIMIZATION FOR EFFICIENTLY IMPROVING GENERALIZATION (Not specified in the paper)
Source: Sharpness-Aware Minimization for Efficiently Improving Generalization (SAM).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | labels (class decisions) (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates SAM on image classification across standard supervised training, finetuning, and noisy-label settings. The modality coverage is image input and class-label output, with no additional task intent (such as generation, detection, or control) explicitly introduced. The OCR text supports a 2D image domain with fixed-size model interfaces in the reported setups, and a fixed 0D class-decision output per example. Attention and state dynamics are inferred as static and direct because the described setup is supervised per-example classification without runtime input-selection mechanisms or persistent task-state construction.

## Evidence
### Task: Image classification
- "In order to assess SAM's efficacy, we apply it to a range of different tasks, including image classification from scratch (including on CIFAR-10, CIFAR-100, and ImageNet), finetuning pretrained models, and learning with noisy labels." (Section 3 EMPIRICAL EVALUATION)
- "To assess SAM's performance at larger scale, we apply it to ResNets (He et al., 2015) of different depths (50, 101, 152) trained on ImageNet (Deng et al., 2009). In this setting, following prior work (He et al., 2015; Szegedy et al., 2015), we resize and crop images to 224-pixel resolution" (Section 3.1 IMAGE CLASSIFICATION FROM SCRATCH)
- "In particular, we measure the effect of applying SAM in the classical noisy-label setting for CIFAR-10, in which a fraction of the training set's labels are randomly flipped; the test set remains unmodified (i.e., clean)." (Section 3.3 ROBUSTNESS TO LABEL NOISE)
- Inference: `Input = images` is directly supported by repeated references to ImageNet/CIFAR image-processing and explicit image resizing/cropping; `In Dimension = 2D (x, y)` and `In Dynamics = Fixed` are inferred from fixed-resolution preprocessing (e.g., "resize and crop images to 224-pixel resolution"). `Output = labels (class decisions)` and `Out Dimension = 0D` are inferred from the paper's classification framing and explicit label handling ("labels are randomly flipped"). `Attention Dynamic = Static` and `State Dynamic = Direct` are inferred because the described task interface is per-example supervised classification without runtime observation selection, retrieval, or persistent constructed state across interactions.
