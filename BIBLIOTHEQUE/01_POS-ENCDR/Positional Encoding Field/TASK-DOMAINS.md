# Positional Encoding Field (Year not specified)
Source: Positional Encoding Field.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Novel view synthesis (single-image) | single input image; target camera pose/viewpoint | 2D (x, y) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | novel-view image (target viewpoint) | 2D (x, y) | Fixed (inferred) |
| Object-level 3D editing | point cloud of the object; original background image | 3D (x, y, z); 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | edited image content (object recomposed with original background) | 2D (x, y) (inferred) | Fixed (inferred) |
| Object removal | image tokens with masked region; noise tokens for fill | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | image content with removed object | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper focuses on single-image novel view synthesis and shows additional spatial editing tasks (object-level 3D editing and object removal) after training. The tasks are primarily image-based (2D), with object-level editing explicitly manipulating a 3D point cloud; fixed interface dynamics are inferred from the model's regular 2D token grid. Attention and state dynamics are not specified in the paper.

## Evidence
### Task: Novel view synthesis (single-image)
- "In this work, we mainly want to leverage these findings to address novel view synthesis (NVS) problem from a single image." (Section 3.1)
- "conditioned on the source reconstruction and target camera pose, we reassign positional encodings so that tokens migrate to their new projected locations." (Section 3.1)
- "In each case, a single input image is provided, and subsequent frames are generated under different target viewpoints." (Section 4.2)
- "where  $x_{src}$  and  $x_{tgt}$  denote the image tokens of the source view with transformed PEs and the target view, respectively" (Section 3.4)
- "noise tokens are placed on a regular 2D grid with depth initialized to zero" (Section 3.4)
- Inference: In/Out Dynamics set to Fixed because tokens are placed on a "regular 2D grid" with a fixed valid grid. (Section 3.4)

### Task: Object-level 3D editing
- "After training, our NVS model acquires the ability to reason over visual tokens in 3D space and generate consistent content." (Section 4.4)
- "we perform object-level 3D editing by isolating the point cloud of the book" (Section 4.4)
- "rotating it to a new viewpoint, and recomposing it with the original background." (Section 4.4)
- "noise tokens are placed on a regular 2D grid with depth initialized to zero" (Section 3.4)
- Inference: The background/image output are treated as 2D (x, y), and In/Out Dynamics are set to Fixed, based on the model's "regular 2D grid." (Section 3.4)

### Task: Object removal
- "we achieve object removal by discarding the tokens corresponding to the masked human region and replenishing them with noise, resulting in a realistic removal effect." (Section 4.4)
- "noise tokens are placed on a regular 2D grid with depth initialized to zero" (Section 3.4)
- Inference: Input/output are treated as 2D (x, y), and In/Out Dynamics are set to Fixed, based on the model's "regular 2D grid." (Section 3.4)
