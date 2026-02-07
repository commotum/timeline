# Multi-Scale Context Aggregation by Dilated Convolutions (Not specified in the paper)
Source: Multi-Scale Context Aggregation by Dilated Convolutions.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| semantic segmentation (pixel-wise labeling) | color images | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | dense label assignments | 2D (x, y) | Not specified in the paper. |

## Summary
The paper targets semantic segmentation as a dense prediction task on 2D images, producing per-pixel label maps. Input and output size dynamics are not explicitly specified, while the convolutional module implies static attention and constructed internal representations.

## Evidence
### Task: semantic segmentation (pixel-wise labeling)
- "semantic segmentation" (Section 1 Introduction)
- "label for each pixel" (Section 1 Introduction)
- "raw image as input" (Section 6 Conclusion)
- "dense label assignments" (Section 6 Conclusion)
- "no pooling or subsampling" (Section 1 Introduction)
- "representation in each layer" (Section 3 Multi-Scale Context Aggregation)
- Inference: Attention Dynamic = Static (inferred) because the module is described as a fixed convolutional stack (see quote above, Section 1 Introduction).
- Inference: State Dynamic = Constructed (inferred) because the paper describes per-layer representations used for prediction (see quote above, Section 3 Multi-Scale Context Aggregation).
