# Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers (Not specified in the paper.)
Source: SETR- Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| semantic segmentation | images | 2D (x, y) | Capped (inferred) | Static (inferred) | Direct (inferred) | pixel-wise semantic labels (segmentation map) | 2D (x, y) | Capped (inferred) |

## Summary
The paper covers one task intent: semantic segmentation for natural images. The task maps 2D image inputs to 2D pixel-wise semantic outputs. Based on the reported resize/crop/sliding-window setup and patchized sequence interface, input and output dynamics are capped (inferred) rather than open. Attention is static (inferred) and state is direct (inferred) from the described full-sequence self-attention encoder and feed-forward segmentation decoding pipeline.

## Evidence
### Task: semantic segmentation
- "In this paper, we aim to provide an alternative perspective by treating semantic segmentation as a sequence-to-sequence prediction task." (Section Abstract)
- "The first layer takes as input the image, denoted as  $H \times W \times 3$  with  $H \times W$ specifying the image size in pixels." (Section 3.1. FCN-based semantic segmentation)
- "As the goal of the decoder is to generate the segmentation results in the original 2D image space  $(H \times W)$" (Section 3.3. Decoder designs)
- Inference: `Capped` input/output dynamics are inferred from bounded preprocessing/evaluation settings ("we apply random resize with ratio between 0.5 and 2, random cropping (768, 512 and 480 for Cityscapes, ADE20K and Pascal Context respectively)" and "Sliding window is adopted for test" in Section 4.1. Experimental setup). `Static` attention is inferred from the fixed full-sequence self-attention computation ("Given the 1D embedding sequence E as input, a pure transformer based encoder is employed" and "This means each transformer layer has a global receptive field" in Section 3.2. Segmentation transformers (SETR)). `Direct` state is inferred because the paper specifies a direct image-to-segmentation pipeline without persistent external memory/state across interactions (Sections 3.2 and 3.3).
