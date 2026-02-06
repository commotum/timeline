# ARC Is a Vision Problem! (Not specified in the paper)
Source: ARC Is a Vision Problem!.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image-to-image translation (per-pixel classification) | Few-shot demo pairs of 2D grids + inference input grid | 2D (x, y) | Capped (inferred) | Static (inferred) | Constructed (inferred) | 2D output grid (color indices) | 2D (x, y) | Capped (inferred) |

## Summary
The paper frames ARC as image-to-image translation (per-pixel classification) on 2D grid inputs and outputs, using few-shot demo pairs to infer the transformation for a new grid. The task domain is a single 2D grid modality with bounded grid sizes (maximum $30 \times 30$), so dynamics are capped. Attention and state dynamics are inferred from the fixed-canvas vision model with per-task task tokens as static attention and constructed state.

## Evidence
### Task: Image-to-image translation (per-pixel classification)
- "Each task, denoted by T, involves a unique underlying transformation rule, mapping from an input x to an output y." (Section 3.1. ARC Problem Definition)
- "At inference time, only the demo pairs  $\mathcal{D}_{\text{demo}}^T$  and one input  $x_{\text{infer}} \in \mathcal{D}_{\text{infer}}^T$  are given," (Section 3.1. ARC Problem Definition)
- "x and y are both 2D grids with maximum size  $30 \times 30$ ," (Section 3.1. ARC Problem Definition)
- "we formulate reasoning on each task as an image-to-image translation problem." (Section 3.2. Image-to-Image Translation)
- "We frame the problem as per-pixel classification, analogous to the semantic segmentation problem [38]." (Section 3.2. Image-to-Image Translation)
- "The task token is represented as a learnable embedding dependent on T." (Section 3.2. Image-to-Image Translation)
- Inference: In/Out Dynamics labeled Capped (inferred) because grids have a stated maximum size and tasks are "2 to 4-shot" (Section 3.1). Attention Dynamic marked Static (inferred) because "the input canvas is divided into non-overlapping patches (e.g.,  $2\times2$ ), projected by a linear embedding" (Section 3.3). State Dynamic marked Constructed (inferred) because the model uses a learned task token per task ("The task token is represented as a learnable embedding dependent on T." in Section 3.2).
