# MaxViT: Multi-Axis Vision Transformer (Not specified in the paper.)
Source: MaxViT- Multi-Axis Vision Transformer.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | class label (inferred) | 0D (inferred) | Fixed (inferred) |
| Object detection (bounding box) | images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | bounding boxes | 2D (x, y) (inferred) | Not specified in the paper. |
| Instance segmentation | images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | instance segmentation masks (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |
| Image aesthetics/quality assessment | images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | score histogram | 1D (t) (inferred) | Fixed (inferred) |
| Unconditional image generation | latent code z | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | images | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper evaluates a single vision backbone across multiple image tasks—classification, object detection, instance segmentation, and aesthetics/quality assessment—and also tests unconditional image generation. Inputs are 2D images at fixed resolutions in the reported experiments, while the generation setup uses a latent code to produce 2D images at 128x128. Attention operates over fixed windows/grids and the backbone is a feedforward stack of repeated blocks, so attention is static and state is direct (inferred); output dynamics for detection/segmentation are not specified.

## Evidence
### Task: Image classification
- "We validated the efficacy of our proposed model on various vision tasks: ImageNet classification [48]" (Section 4 Experiments)
- "ImageNet-1K. We show in Table 2 the performance comparisons on ImageNet-1K classification." (Section 4.1)
- Inference: In Dimension and In Dynamics inferred from "Under the basic 224×224 setting" (Section 4.1). Output/Out Dimension/Out Dynamics inferred from "followed by the final classification head." (Appendix A.1). Attention Static inferred from "partitioning into non-overlapping windows, each of size  $P \times P$ " and "using a fixed  $G \times G$  uniform grid" (Section 3.2). State Direct inferred from "hierarchically stacking repeated blocks composed of Max-SA and convolutions." (Section 3 Method)

### Task: Object detection (bounding box)
- "image object detection and instance segmentation [53]" (Section 4 Experiments)
- "We evaluated the MaxViT architectures on the COCO2017 [53] object bounding box detection and instance segmentation tasks" (Section 4.2)
- "For both tasks, the input images are resized to  $896 \times 896$ ." (Section B.2)
- Inference: In Dimension and In Dynamics inferred from the fixed resize in Section B.2. Out Dimension inferred from "object bounding box detection" (Section 4.2). Attention Static inferred from "partitioning into non-overlapping windows, each of size  $P \times P$ " and "using a fixed  $G \times G$  uniform grid" (Section 3.2). State Direct inferred from "hierarchically stacking repeated blocks composed of Max-SA and convolutions." (Section 3 Method)

### Task: Instance segmentation
- "image object detection and instance segmentation [53]" (Section 4 Experiments)
- "In the instance segmentation task, a well-known Cascade Mask-RCNN framework [28] was employed." (Section 4.2)
- "For both tasks, the input images are resized to  $896 \times 896$ ." (Section B.2)
- Inference: In Dimension and In Dynamics inferred from the fixed resize in Section B.2. Output/Out Dimension inferred from "instance segmentation task" (Section 4.2). Attention Static inferred from "partitioning into non-overlapping windows, each of size  $P \times P$ " and "using a fixed  $G \times G$  uniform grid" (Section 3.2). State Direct inferred from "hierarchically stacking repeated blocks composed of Max-SA and convolutions." (Section 3 Method)

### Task: Image aesthetics/quality assessment
- "image aesthetics/quality assessment [61]" (Section 4 Experiments)
- "We trained and evaluated the MaxViT model on the AVA benchmark [61]." (Section B.3)
- "Each image in the dataset has a histogram of scores associated with it" (Section B.3)
- Inference: In Dimension and In Dynamics inferred from "input resolutions:  $224 \times 224$ ,  $384 \times 384$  and  $512 \times 512$ ." (Section B.3). Out Dimension and Out Dynamics inferred from "histogram of scores" and "10 neurons followed by softmax." (Sections B.3, A.3). Attention Static inferred from "partitioning into non-overlapping windows, each of size  $P \times P$ " and "using a fixed  $G \times G$  uniform grid" (Section 3.2). State Direct inferred from "hierarchically stacking repeated blocks composed of Max-SA and convolutions." (Section 3 Method)

### Task: Unconditional image generation
- "and unconditional image generation [26]." (Section 4 Experiments)
- "We evaluate the generative ability of MaxViT blocks to generate images of 128x128 resolution on ImageNet-1K." (Section 4.4)
- "MaxViT-GAN first takes a latent code  $z \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$  as input" (Appendix A.4)
- Inference: In Dimension and In Dynamics inferred from the latent code input in Appendix A.4. Out Dimension and Out Dynamics inferred from "generate images of 128x128 resolution" (Section 4.4). Attention Static inferred from "partitioning into non-overlapping windows, each of size  $P \times P$ " and "using a fixed  $G \times G$  uniform grid" (Section 3.2). State Direct inferred from "hierarchically stacking repeated blocks composed of Max-SA and convolutions." (Section 3 Method)
