## 1. Basic Metadata

Title: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (Title)
Authors: "Ze Liu<sup>1,2†\*</sup> Yutong Lin<sup>1,3†\*</sup> Yue Cao<sup>1\*</sup> Han Hu<sup>1\*‡</sup> Yixuan Wei<sup>1,4†</sup>" (Title block) "Zheng Zhang<sup>1</sup> Stephen Lin<sup>1</sup> Baining Guo<sup>1</sup>" (Title block)
Year: Year not specified.
Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

"This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision." (Abstract)

## 3. Tasks Evaluated

- Task name: Image classification
  - Task type: Classification
  - Dataset(s) used: ImageNet-1K; ImageNet-22K (pre-training setting)
  - Domain: RGB images
  - Quotes: "We conduct experiments on ImageNet-1K image classification [18], COCO object detection [39], and ADE20K semantic segmentation [74]." (Section 4) "For image classification, we benchmark the proposed Swin Transformer on ImageNet-1K [18], which contains 1.28M training images and 50K validation images from 1,000 classes." (Section 4.1) "We also pre-train on the ImageNet-22K dataset, which contains 14.2 million images and 22K classes." (Section 4.1) "It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT." (Section 3.1)

- Task name: Object detection
  - Task type: Detection
  - Dataset(s) used: COCO 2017
  - Domain: RGB images
  - Quotes: "We conduct experiments on ImageNet-1K image classification [18], COCO object detection [39], and ADE20K semantic segmentation [74]." (Section 4) "Object detection and instance segmentation experiments are conducted on COCO 2017, which contains" (Section 4.2) "118K training, 5K validation and 20K test-dev images." (Section 4.2) "It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT." (Section 3.1)

- Task name: Instance segmentation
  - Task type: Segmentation
  - Dataset(s) used: COCO 2017
  - Domain: RGB images
  - Quotes: "Object detection and instance segmentation experiments are conducted on COCO 2017, which contains" (Section 4.2) "118K training, 5K validation and 20K test-dev images." (Section 4.2) "It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT." (Section 3.1)

- Task name: Semantic segmentation
  - Task type: Segmentation
  - Dataset(s) used: ADE20K
  - Domain: RGB images
  - Quotes: "We conduct experiments on ImageNet-1K image classification [18], COCO object detection [39], and ADE20K semantic segmentation [74]." (Section 4) "ADE20K [74] is a widely-used semantic segmentation dataset, covering a broad range of 150 semantic categories. It has 25K images in total, with 20K for training, 2K for validation, and another 3K for testing." (Section 4.3) "It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT." (Section 3.1)

## 4. Domain and Modality Scope

- Evaluation scope: Multiple datasets within the same modality (RGB images). Evidence: "We conduct experiments on ImageNet-1K image classification [18], COCO object detection [39], and ADE20K semantic segmentation [74]." (Section 4) "It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT." (Section 3.1)
- Single domain vs multiple domains: The paper stays within image inputs; explicit cross-domain claims are not stated. Evidence: "It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT." (Section 3.1)
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image classification (ImageNet-1K/22K) | Not explicitly stated across tasks. | Yes (explicit fine-tuning setting for ImageNet-22K to ImageNet-1K). | Not specified. | "We also pre-train on the ImageNet-22K dataset, which contains 14.2 million images and 22K classes." (Section 4.1) "In ImageNet-1K fine-tuning, we train for 30 epochs with a batch size of 1024, a constant learning rate of 10<sup>-5</sup>, and a weight decay of 10<sup>-8</sup>." (Section 4.1) |
| Object detection (COCO 2017) | Not explicitly stated across tasks. | Not explicitly stated; pre-trained initialization is used. | Not specified; task-specific frameworks are used. | "For the ablation study, we consider four typical object detection frameworks: Cascade Mask R-CNN [26, 6], ATSS [71], RepPoints v2 [12], and Sparse RCNN [52] in mmdetection [10]." (Section 4.2) "For system-level comparison, we adopt an improved HTC [9] (denoted as HTC++) with instaboost [20], stronger multi-scale training [7], 6x schedule (72 epochs), soft-NMS [5], and ImageNet-22K pre-trained model as initialization." (Section 4.2) |
| Instance segmentation (COCO 2017) | Not explicitly stated across tasks. | Not explicitly stated; pre-trained initialization is used. | Not specified; task-specific frameworks are used. | "Object detection and instance segmentation experiments are conducted on COCO 2017, which contains" (Section 4.2) "118K training, 5K validation and 20K test-dev images." (Section 4.2) "For system-level comparison, we adopt an improved HTC [9] (denoted as HTC++) with instaboost [20], stronger multi-scale training [7], 6x schedule (72 epochs), soft-NMS [5], and ImageNet-22K pre-trained model as initialization." (Section 4.2) |
| Semantic segmentation (ADE20K) | Not explicitly stated across tasks. | Not explicitly stated; some models are pre-trained on ImageNet-22K. | Not specified; task-specific framework is used. | "We utilize UperNet [63] in mmseg [16] as our base framework for its high efficiency." (Section 4.3) "‡ indicates that the model is pre-trained on ImageNet-22K." (Table 3 caption) |

## 6. Input and Representation Constraints

- Input modality: "It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT." (Section 3.1)
- Fixed patch size and patch features: "In our implementation, we use a patch size of  $4\times 4$  and thus the feature dimension of each patch is  $4\times 4\times 3=48$ ." (Section 3.1)
- Token count tied to image resolution: 'The Transformer blocks maintain the number of tokens ( $\frac{H}{4} \times \frac{W}{4}$ ), and together with the linear embedding are referred to as "Stage 1".' (Section 3.1)
- Hierarchical patch merging/downsampling: "The first patch merging layer concatenates the features of each group of  $2\times 2$  neighboring patches, and applies a linear layer on the 4C-dimensional concatenated features. This reduces the number of tokens by a multiple of  $2\times 2=4$  ( $2\times$  downsampling of resolution), and the output dimension is set to 2C." (Section 3.1)
- Fixed window size for attention: "where the former is quadratic to patch number hw, and the latter is linear when M is fixed (set to 7 by default)." (Section 3.2)
- Padding requirement: "To make the window size (M,M) divisible by the feature map size of (h,w), bottom-right padding is employed on the feature map if needed." (Section 4.1)
- Input resizing for detection: "multi-scale training [8, 52] (resizing the input such that the shorter side is between 480 and 800 while the longer side is at most 1333)" (Section 4.2)

## 7. Context Window and Attention Structure

- Maximum sequence length (per attention window): "Supposing each window contains  $M \times M$  patches" (Section 3.2) and "where the former is quadratic to patch number hw, and the latter is linear when M is fixed (set to 7 by default)." (Section 3.2)
- Fixed vs variable sequence length: Per-window length is fixed by M; total token count depends on image resolution. Evidence: 'The Transformer blocks maintain the number of tokens ( $\frac{H}{4} \times \frac{W}{4}$ ), and together with the linear embedding are referred to as "Stage 1".' (Section 3.1) "Supposing each window contains  $M \times M$  patches" (Section 3.2)
- Attention type: Hierarchical and windowed with shifted windows. Evidence: "we propose a hierarchical Transformer whose representation is computed with Shifted windows." (Abstract) "For efficient modeling, we propose to compute self-attention within local windows. The windows are arranged to evenly partition the image in a non-overlapping manner." (Section 3.2) "To introduce cross-window connections while maintaining the efficient computation of non-overlapping windows, we propose a shifted window partitioning approach which alternates between two partitioning configurations in consecutive Swin Transformer blocks." (Section 3.2)
- Mechanisms to manage computational cost: Windowing, fixed window size, and efficient shifted computation with masking. Evidence: "The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection." (Abstract) "The number of patches in each window is fixed, and thus the complexity becomes linear to image size." (Section 1) "Here, we propose a more efficient batch computation approach by cyclic-shifting toward the top-left direction, as illustrated in Figure 4. After this shift, a batched window may be composed of several sub-windows that are not adjacent in the feature map, so a masking mechanism is employed to limit self-attention computation to within each sub-window." (Section 3.2)

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Relative position bias. Evidence: "including a relative position bias  $B \in \mathbb{R}^{M^2 \times M^2}$  to each head in computing similarity:" (Section 3.2)
- Where it is applied: As an attention bias term in the similarity computation. Evidence: "Attention(Q, K, V) = SoftMax(QK^{T}/\sqrt{d} + B)V, \quad (4)" (Section 3.2)
- Fixed vs modified/ablated: Compared against alternatives; absolute position embedding is not adopted. Evidence: "We observe significant improvements over counterparts without this bias term or that use absolute position embedding, as shown in Table 4. Further adding absolute position embedding to the input as in [19] drops performance slightly, thus it is not adopted in our implementation." (Section 3.2) "Table 4 shows comparisons of different position embedding approaches." (Section 4.4)

## 9. Positional Encoding as a Variable

- Role: Treated as an ablated design choice. Evidence: "Table 4 shows comparisons of different position embedding approaches." (Section 4.4)
- Multiple positional encodings compared: "Swin-T with relative position bias yields +1.2%/+0.8% top-1 accuracy on ImageNet-1K, +1.3/+1.5 box AP and +1.1/+1.3 mask AP on COCO, and +2.3/+2.9 mIoU on ADE20K in relation to those without position encoding and with absolute position embedding, respectively, indicating the effectiveness of the relative position bias." (Section 4.4)
- Claim about criticality: The paper reports that relative position bias improves results and absolute position embedding can reduce performance; no claim that PE is secondary. Evidence: "Further adding absolute position embedding to the input as in [19] drops performance slightly, thus it is not adopted in our implementation." (Section 3.2)

## 10. Evidence of Constraint Masking

- Model size scaling: "We also introduce Swin-T, Swin-S and Swin-L, which are versions of about  $0.25\times, 0.5\times$  and  $2\times$  the model size and computational complexity, respectively." (Section 3.3)
- Dataset sizes: "ImageNet-1K [18], which contains 1.28M training images and 50K validation images from 1,000 classes." (Section 4.1) "Object detection and instance segmentation experiments are conducted on COCO 2017, which contains" (Section 4.2) "118K training, 5K validation and 20K test-dev images." (Section 4.2) "ADE20K [74] is a widely-used semantic segmentation dataset, covering a broad range of 150 semantic categories. It has 25K images in total, with 20K for training, 2K for validation, and another 3K for testing." (Section 4.3)
- Data scaling gains: "For Swin-B, the ImageNet-22K pre-training brings 1.8%~1.9% gains over training on ImageNet-1K from scratch." (Section 4.1)
- Architectural gains (shifted windows): "Swin-T with the shifted window partitioning outperforms the counterpart built on a single window partitioning at each stage by +1.1% top-1 accuracy on ImageNet-1K, +2.8 box AP/+2.2 mask AP on COCO, and +2.8 mIoU on ADE20K." (Section 4.4)

## 11. Architectural Workarounds

- Windowed attention: "For efficient modeling, we propose to compute self-attention within local windows. The windows are arranged to evenly partition the image in a non-overlapping manner." (Section 3.2)
- Shifted windows: "To introduce cross-window connections while maintaining the efficient computation of non-overlapping windows, we propose a shifted window partitioning approach which alternates between two partitioning configurations in consecutive Swin Transformer blocks." (Section 3.2)
- Hierarchical stages via patch merging: "Swin Transformer constructs a hierarchical representation by starting from small-sized patches (outlined in gray) and gradually merging neighboring patches in deeper Transformer layers." (Section 1) "The first patch merging layer concatenates the features of each group of  $2\times 2$  neighboring patches, and applies a linear layer on the 4C-dimensional concatenated features." (Section 3.1)
- Masking for shifted windows: "a masking mechanism is employed to limit self-attention computation to within each sub-window." (Section 3.2)
- Padding to align window size: "To make the window size (M,M) divisible by the feature map size of (h,w), bottom-right padding is employed on the feature map if needed." (Section 4.1)

## 12. Explicit Limitations and Non-Claims

- Limitation: "A thorough kernel optimization is beyond the scope of this paper." (Section 4.2)
- Other limitations or non-claims about open-world, unrestrained multi-task learning, or meta-learning: Not stated.

### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: Single modality (RGB images) across multiple datasets. Evidence: "It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT." (Section 3.1) "We conduct experiments on ImageNet-1K image classification [18], COCO object detection [39], and ADE20K semantic segmentation [74]." (Section 4)
- Task structure: Multiple supervised vision tasks evaluated separately (classification, detection/instance segmentation, semantic segmentation). Evidence: "We conduct experiments on ImageNet-1K image classification [18], COCO object detection [39], and ADE20K semantic segmentation [74]." (Section 4)
- Representation rigidity: Fixed patch size and fixed window size; hierarchical downsampling. Evidence: "In our implementation, we use a patch size of  $4\times 4$  and thus the feature dimension of each patch is  $4\times 4\times 3=48$ ." (Section 3.1) "where the former is quadratic to patch number hw, and the latter is linear when M is fixed (set to 7 by default)." (Section 3.2)
- Model sharing vs specialization: Task-specific frameworks and pre-trained initialization are used; no explicit joint multi-task training. Evidence: "For the ablation study, we consider four typical object detection frameworks: Cascade Mask R-CNN [26, 6], ATSS [71], RepPoints v2 [12], and Sparse RCNN [52] in mmdetection [10]." (Section 4.2) "We utilize UperNet [63] in mmseg [16] as our base framework for its high efficiency." (Section 4.3)
- Role of positional encoding: Relative position bias is chosen after ablation against alternatives. Evidence: "Table 4 shows comparisons of different position embedding approaches." (Section 4.4) "We observe significant improvements over counterparts without this bias term or that use absolute position embedding, as shown in Table 4." (Section 3.2)

## 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple tasks within the image modality: "We conduct experiments on ImageNet-1K image classification [18], COCO object detection [39], and ADE20K semantic segmentation [74]." (Section 4) Inputs are consistently images, e.g., "It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT." (Section 3.1), and there is no claim of cross-domain or multi-modality evaluation.
