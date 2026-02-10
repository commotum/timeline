# Swin Transformer V2: Scaling Up Capacity and Resolution (2022)
Source: Swin Transformer V2- Scaling Up Capacity and Resolution.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | Images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Class labels (top-1) (inferred) | 0D (inferred) | Fixed (inferred) |
| Object detection | Images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Bounding boxes and instance masks (inferred) | 2D (x, y) (inferred) | Capped (inferred) |
| Semantic segmentation | Images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Pixel-wise semantic labels (inferred) | 2D (x, y) (inferred) | Capped (inferred) |
| Video action classification | Video clips | 3D (x, y, t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Action class labels (top-1) (inferred) | 0D (inferred) | Fixed (inferred) |
| Self-supervised pre-training (SimMIM) | Images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper reports Swin Transformer V2 on four benchmarked downstream tasks (image classification, object detection, semantic segmentation, and video action classification) plus an explicit self-supervised pre-training phase (SimMIM). Inputs span 2D images and 3D spatiotemporal video clips, while downstream outputs include 0D class labels and 2D dense/object-level predictions. Based on the OCR text, interfaces are best supported as Capped, with Static attention and Direct state processing; the self-supervised output interface is not explicitly specified in this OCR.

## Evidence
### Task: Image classification
- "It set new performance records on 4 representative vision tasks, including ImageNet-V2 image classification, COCO object detection, ADE20K semantic segmentation, and Kinetics-400 video action classification." (Abstract)
- "Image classification. ImageNet-1K V1 and V2 val are used [18,55] for evaluation." (Section 4.1)
- "In evaluation, we test top-1 accuracy on both ImageNet-1K V1 and V2." (Section A2.2)
- Inference: `2D (x, y)` input and `0D` output are inferred from image input plus single-label top-1 reporting; `Capped` is inferred from bounded resolutions (e.g., "We consider input image sizes of  $256 \times 256$  and  $384 \times 384$ ." in Section A2.1 and "We adopt an input image size of  $640 \times 640$  for experiments." in Section A2.2). `Static` attention is inferred from predefined attention windows ("the window size can be either fixed or changed during finetuning." in Section 2 and `M^2` patches per window in Section 3.1), and `Direct` state is inferred because no persistent external memory/state mechanism is described.

### Task: Object detection
- "It set new performance records on 4 representative vision tasks, including ImageNet-V2 image classification, COCO object detection, ADE20K semantic segmentation, and Kinetics-400 video action classification." (Abstract)
- "- Object detection. COCO [44] is used for evaluation." (Section 4.1)
- "63.1 / 54.4 box / mask AP on the COCO test-dev set of object detection" (Section 1)
- Inference: `2D (x, y)` input/output and "Bounding boxes and instance masks" output are inferred from COCO detection reporting with box/mask AP. `Capped` is inferred from bounded training/eval sizes (e.g., "the input image resolution is set  $1536\times1536$  with a multi-scale ratio of [0.1,2.0]." in Section A2.2). `Static` attention and `Direct` state are inferred from the same Swin attention-window formulation used across tasks (Sections 2 and 3.1) and no explicit persistent state mechanism.

### Task: Semantic segmentation
- "It set new performance records on 4 representative vision tasks, including ImageNet-V2 image classification, COCO object detection, ADE20K semantic segmentation, and Kinetics-400 video action classification." (Abstract)
- "- Semantic segmentation. ADE20K [85] is used." (Section 4.1)
- "This suggests scaling up vision model is beneficial for pixel-level vision recognition tasks." (Section 4.2)
- Inference: output is inferred as pixel-wise semantic labels on a `2D (x, y)` grid from the task name and "pixel-level" description. `Capped` is inferred from bounded image/window settings (e.g., "The input image size (window size) is set  $640 \times 640$  ( $40 \times 40$ )." in Section A2.2). `Static` attention and `Direct` state are inferred from the predefined windowed-attention architecture and no explicit persistent state construction described in OCR text.

### Task: Video action classification
- "It set new performance records on 4 representative vision tasks, including ImageNet-V2 image classification, COCO object detection, ADE20K semantic segmentation, and Kinetics-400 video action classification." (Abstract)
- "- *Video action classification*. Kinetics-400 (K400) [37] is used in evaluation." (Section 4.1)
- "In the first stage, an input resolution of  $256\times256\times8$  with  $16\times16\times8$  window size is adopted." (Section A2.2)
- "It achieves 86.8% top-1 accuracy" (Section 4.2)
- Inference: input is inferred as `3D (x, y, t)` from explicit `256x256x8` and `320x320x8` video clips; output is inferred as `0D` class label from top-1 accuracy reporting. `Capped` is inferred from bounded clip sizes/window sizes. `Static` attention and `Direct` state are inferred from fixed/selected attention-window processing and no OCR evidence of persistent external memory/state.

### Task: Self-supervised pre-training (SimMIM)
- "3) A self-supervised pretraining method, SimMIM, to reduce the needs of vast labeled images." (Abstract)
- "Stage-1 self-supervised pre-training The model is first pre-trained using a self-supervised learning approach [1] on the ImageNet-22K-ext dataset (70 million images) for 20 epochs." (Section A2.2)
- "[1] Anonymous. Simmim: A simple framework for masked image modeling. In *CVPR submission*, 2022." (References)
- Inference: input is inferred as `2D (x, y)` and `Capped` from explicit image-based pre-training at bounded resolution (e.g., "we adopt a smaller image size of  $192 \times 192$ ." in Section A2.2). `Static` attention and `Direct` state are inferred from the same Swin windowed-attention architecture. Output/Out Dimension/Out Dynamics are marked "Not specified in the paper." because this OCR does not explicitly describe the pre-training target interface.
