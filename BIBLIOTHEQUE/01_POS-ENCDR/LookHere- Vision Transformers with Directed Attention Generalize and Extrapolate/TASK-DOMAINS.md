# LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate (Year not specified in the paper)
Source: LookHere- Vision Transformers with Directed Attention Generalize and Extrapolate.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | class probabilities | 0D (inferred) | Fixed (inferred) |
| Semantic segmentation | images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | segmentation map | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper evaluates image classification on multiple ImageNet test sets, including adversarial-attack and calibration evaluations, and semantic segmentation via finetuning, linear probing, and patch logit-lens on ADE20k/Cityscapes/ImageNet-S. Inputs are 2D images and outputs are class probabilities or 2D segmentation maps, with resolutions varying across experiments (e.g., 224^2 to 1024^2 for classification and 512^2/768^2 for segmentation), indicating capped spatial sizes. Attention is constrained by fixed FOV masks and the model operates as a direct mapping from patch tokens to outputs.

## Evidence
### Task: Image classification
- "We demonstrate that LookHere improves performance on classification (avg. \( \gamma \) 1.6\%), against adversarial attack (avg. $\uparrow 5.4\%$ ), and decreases calibration error (avg. $\downarrow 1.5\%$ ) — on ImageNet without extrapolation." (Abstract)
- "We test all 80 models on six ImageNet test sets." (Section 4.1 Setup)
- "Adversarial Attacks. We perform Fast Gradient Sign Method (FGSM [82]) adversarial attacks with two strengths  $(\frac{1}{255}, \frac{3}{255})$  on all models using Val images." (Section 4.1 Setup)
- "Calibration Estimates. We calculate the Expected Calibration Error (ECE [83]) with 15 bins of all models using Val images." (Section 4.1 Setup)
- "Extrapolating. With the best model per method, we test on images larger than  $224^2$  px, increasing the number of patches and we test on images smaller than  $224^2$  px, decreasing the number of patches; for both experiments, no further training is performed — the models are tested on their resolution generalization ability." (Section 4.1 Setup)
- "Figure 1: ViT-B/16 models trained for 150 epochs on ImageNet at  $224^2$  px and tested up to  $1024^2$  px." (Figure 1)
- "- CLS token with an MLP classifying head final linear layer weights are initialized to 0 and biases to -6.9 (so all class probabilities start at  $\frac{1}{1000}$ )" (Section A.4.1 Training ViTs)
- "We introduce 2D attention masks that assign each attention head a direction and a FOV, preventing attention outside the head's FOV." (Section 3 LookHere, Design Motivation)
- "A ViT splits an image into a grid of non-overlapping patches, flattens the grid into a sequence, and flattens the patches into vectors;" (Section 2 Background and Related Work)
- Inference: Input is treated as 2D and capped because images are evaluated at multiple fixed resolutions (e.g., 224^2 px and tested up to 1024^2 px); attention is static due to fixed FOV masks; state is direct because the model maps patch sequences through transformer layers; output dimensionality/dynamics are point-like and fixed because it produces class probabilities over 1000 classes. (Abstract; Section 4.1 Setup; Figure 1; Section 3 LookHere, Design Motivation; Section 2 Background and Related Work; Section A.4.1 Training ViTs)

### Task: Semantic segmentation
- "Segmentation. With the best model per method, we finetune following the Segmenter protocol with a linear decoder [84]." (Section 4.1 Setup)
- "Additionally, we probe the patches by only training a linear layer to produce a low-resolution logit map which is upsampled to obtain a full resolution segmentation map, following [85]." (Section 4.1 Setup)
- "We run these experiments on ADE20k [86] at 512<sup>2</sup> px and Cityscapes [87] at 768<sup>2</sup> px." (Section 4.1 Setup)
- "We leverage the ImageNet-S dataset [91], which contains partial segmentation maps for 12k images from Val, covering 919 ImageNet classes." (Section 4.1 Setup)
- Inference: Input/output are treated as 2D and capped because segmentation experiments are run at fixed image resolutions and produce full-resolution segmentation maps; attention is static due to fixed FOV masks and state is direct because the model maps patch sequences through transformer layers. (Section 4.1 Setup; Section 3 LookHere, Design Motivation; Section 2 Background and Related Work)
