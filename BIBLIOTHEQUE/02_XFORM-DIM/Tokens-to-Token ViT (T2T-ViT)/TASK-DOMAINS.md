# Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet (Year not specified in the paper.)
Source: Tokens-to-Token ViT (T2T-ViT).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image classification | images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | class label | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates a single task intent: image classification, trained on ImageNet and transferred to CIFAR10/CIFAR100. The supported modality is image input, which maps to a 2D (x, y) task domain and a 0D class-label output. The interface is fixed-size in practice (fixed image resolutions and fixed token length before the backbone). Attention behavior is static at runtime over the predefined token sequence, while state is constructed through iterative token/feature transformations.

## Evidence
### Task: image classification
- "We conduct the following experiments with T2T-ViT for image classification on ImageNet." (Section 4)
- "We transfer our pretrained T2T-ViT to downstream datasets such as CIFAR10 and CIFAR100." (Section 4.1, Transfer learning)
- "Throughout the experiments on ImageNet, we set default image size as  $224 \times 224$  except for some specific cases on  $384 \times 384$" (Section 4.1)
- "After the final iteration, the output tokens  $T_f$  of the T2T module has fixed length" (Section 3.1)
- "where E is Sinusoidal Position Embedding, LN is layer normalization, fc is one fully-connected layer for classification and y is the output prediction." (Section 3.2)
- Inference: `In Dimension = 2D (x, y)` is inferred from image inputs and explicit height/width structure ("...set default image size as  $224 \times 224$ ...", Section 4.1). `In Dynamics = Fixed` is inferred from fixed image-size settings and fixed token length before the backbone (Sections 4.1 and 3.1). `Attention Dynamic = Static` is inferred because self-attention is applied to the predefined token sequence from the image ("given a sequence of tokens T ... transformed by the self-attention block", Section 3.1), without runtime input-source selection. `State Dynamic = Constructed` is inferred from iterative token/state transformations (`T_i -> T_i' -> I_i -> T_{i+1}` in Eqn. (4), Section 3.1). `Out Dimension = 0D` and `Out Dynamics = Fixed` are inferred from single-label classification output (`fc ... for classification and y is the output prediction`, Section 3.2).
