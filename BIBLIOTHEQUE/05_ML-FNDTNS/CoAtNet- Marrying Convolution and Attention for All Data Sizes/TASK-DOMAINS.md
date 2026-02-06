# CoAtNet: Marrying Convolution and Attention for All Data Sizes (Not specified in the paper.)
Source: CoAtNet- Marrying Convolution and Attention for All Data Sizes.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image classification | images | 2D (x, y) | Fixed (inferred) | Static (inferred) | Direct (inferred) | class labels (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates CoAtNet on image classification, using image datasets such as ImageNet-1K/21K and JFT, which implies 2D image inputs and label outputs. The experimental protocol fixes input resolution per run (e.g., 224 and larger finetuning sizes), so input and output dynamics are treated as fixed, with single-label 0D outputs. Attention and state dynamics are inferred as static/direct because attention is defined over the full spatial grid and the model is described with a standard classification head without any explicit constructed state.

## Evidence
### Task: image classification
- "Our experiments focus on image classification." (Section 4.1 Experiment Setting, Evaluation Protocol)
- "To implement the pre-norm version of relative attention in Eqn. 3 for 2D images of size  $[H \times W]$ ," (Appendix A.1 Model Details, 2D Relative Attention)
- "Instead of adding an additional <cls> token as in ViT to perform classification, we apply global average pooling to the last-stage output to get the representation for simplicity." (Appendix A.1 Model Details, Classification head)
- "we first pre-train our models on each of the three datasets at resolution 224 for 300, 90 and 14 epochs respectively." (Section 4.1 Experiment Setting, Evaluation Protocol)
- "self-attention allows the receptive field to be the entire spatial locations and computes the weights based on the re-normalized pairwise similarity between the pair  $(x_i, x_j)$" (Section 2.1 Merging Convolution and Self-Attention)
- Inference: Input/Output dynamics and 0D label outputs are marked as fixed/0D because the paper trains at a fixed resolution ("pre-train ... at resolution 224") and frames the task as classification; Attention Dynamic is marked Static because attention spans a fixed global spatial space ("receptive field ... entire spatial locations"); State Dynamic is marked Direct because the paper only describes a standard classification head and no constructed memory or state beyond the input.
