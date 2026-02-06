# AN IMAGE IS WORTH 16x16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE (Not specified in the paper.)
Source: An Image is Worth 16×16 Words- Transformers for Image Recognition at Scale (ViT).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image classification | images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | class prediction (label) | 0D (inferred) | Fixed (inferred) |
| masked patch prediction (self-supervised) | image patches (masked patch embeddings) | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | mean color class per corrupted patch (512 colors) | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper covers supervised image classification/recognition across multiple image benchmarks and a self-supervised masked patch prediction objective. Inputs are 2D images (patchified) with outputs that are either a single class prediction (0D) or per-patch color predictions (2D); sequence length can vary with resolution but is bounded by memory, so dynamics are capped. Attention and state characterizations are inferred from the standard Transformer encoder with global self-attention and a feedforward classification/prediction head.

## Evidence
### Task: image classification
- "a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks." (Abstract)
- "We train the model on image classification in supervised fashion." (Introduction)
- "The output of this token is then transformed into a class prediction via a small multi-layer perceptron (MLP)" (Section D.3 Head Type and Class Token)
- Inference: In/Out dimensions inferred from "To handle 2D images, we reshape the image" (Section 3.1 Vision Transformer) and the class prediction output; capped input dynamics inferred from "The Vision Transformer can handle arbitrary sequence lengths (up to memory constraints)" (Section 3.2 Fine-Tuning and Higher Resolution). Attention Static and State Direct inferred from "the self-attention layers are global" (Section 3.1 Inductive bias) and use of a "standard Transformer encoder" (Figure 1 caption).

### Task: masked patch prediction (self-supervised)
- "We employ the *masked patch prediction* objective for preliminary self-supervision experiments." (Section B.1.2 Self-supervision)
- "we corrupt 50% of patch embeddings" (Section B.1.2 Self-supervision)
- "we predict the 3-bit, mean color (i.e., 512 colors in total) of every corrupted patch" (Section B.1.2 Self-supervision)
- Inference: In/Out dimensions inferred from patchified 2D image inputs ("To handle 2D images, we reshape the image" (Section 3.1 Vision Transformer)) and per-patch predictions; capped dynamics inferred from "The Vision Transformer can handle arbitrary sequence lengths (up to memory constraints)" (Section 3.2). Attention Static and State Direct inferred from "the self-attention layers are global" (Section 3.1 Inductive bias) and use of a "standard Transformer encoder" (Figure 1 caption).
