## 1. Basic Metadata

- Title: "Twins: Revisiting the Design of Spatial Attention in Vision Transformers" (Title line).
- Authors: "Xiangxiang Chu<sup>1</sup>, Zhi Tian<sup>2</sup>, Yuqing Wang<sup>1</sup>, Bo Zhang<sup>1</sup>, Haibing Ren<sup>1</sup>, Xiaolin Wei<sup>1</sup>, Huaxia Xia<sup>1</sup>, Chunhua Shen<sup>2</sup>\*" (Title block).
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

Revisits spatial attention design and proposes Twins-PCPVT and Twins-SVT architectures for vision tasks (Abstract: “we revisit the design of the spatial attention” and “we propose two vision transformer architectures, namely, Twins-PCPVT and Twins-SVT”).

## 3. Tasks Evaluated

- Task name: Image classification; Task type: Classification; Dataset(s) used: ImageNet-1K; Domain: images; Evidence: “image-level classification as well as dense detection and segmentation.” (Abstract) and “We report the classification results on ImageNet-1K [39] in Table 1.” (Section 4.1 Classification on ImageNet-1K).
- Task name: Semantic segmentation; Task type: Segmentation; Dataset(s) used: ADE20K; Domain: images (scene parsing); Evidence: “We test on the ADE20K dataset [42], a challenging scene parsing task for semantic segmentation, which is popularly evaluated by recent Transformer-based methods.” (Section 4.2 Semantic Segmentation on ADE20K).
- Task name: Object detection; Task type: Detection; Dataset(s) used: COCO 2017; Domain: images; Evidence: “Specifically, we report standard 1×-schedule (12 epochs) detection results on the COCO 2017 dataset [48] in Tables 3 and 4.” (Section 4.3 Object Detection and Segmentation on COCO).
- Task name: Instance segmentation (object segmentation); Task type: Segmentation; Dataset(s) used: COCO 2017; Domain: images; Evidence: “For  $1\times$  object segmentation with the Mask R-CNN framework, Twins-PCPVT-S brings similar improvements (+2.5% mAP) over PVT-Small.” (Section 4.3 Object Detection and Segmentation on COCO).

## 4. Domain and Modality Scope

- Is evaluation performed on a single domain? Yes — visual image tasks; Evidence: “visual tasks including image-level classification as well as dense detection and segmentation.” (Abstract).
- Is evaluation performed on multiple domains within the same modality? Multiple image datasets are used (ImageNet-1K, ADE20K, COCO); Evidence: “We report the classification results on ImageNet-1K [39]” (Section 4.1), “We test on the ADE20K dataset [42]” (Section 4.2), and “COCO 2017 dataset [48]” (Section 4.3).
- Is evaluation performed on multiple modalities? Not evaluated; the paper notes transformers can process “multi-modality input data including images, videos, texts, speech signals, and point clouds.” (Section 1 Introduction).
- Does the paper claim domain generalization or cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image classification (ImageNet-1K) | Not specified (trained for classification) | Not specified | GAP used for classification head | “We first present the ImageNet classification results with our proposed models.” (Section 4.1 Classification on ImageNet-1K); “For image-level classification, following CPVT, we remove the class token and use global average pooling (GAP) at the end of the stage [9].” (Section 3.1 Twins-PCPVT) |
| Semantic segmentation (ADE20K) | Yes (ImageNet-1k pretraining) | Yes (trained on ADE20K) | Yes (Semantic FPN / UperNet frameworks) | “All models are pretrained on the ImageNet-1k dataset.” (Section 4.2 Semantic Segmentation on ADE20K); “we use the Semantic FPN framework [43]” (Section 4.2); “Swin evaluates its performance using the UperNet framework [44]. We transfer our method to this framework” (Section 4.2) |
| Object detection (COCO) | Not specified | Not specified | Yes (RetinaNet framework) | “We evaluate the performance of our method using two representative frameworks: RetinaNet [46] and Mask RCNN [47]. Specifically, we use our transformer models to build the backbones of these detectors.” (Section 4.3 Object Detection and Segmentation on COCO) |
| Instance segmentation (COCO) | Not specified | Not specified | Yes (Mask R-CNN framework) | “For  $1\times$  object segmentation with the Mask R-CNN framework, Twins-PCPVT-S brings similar improvements (+2.5% mAP) over PVT-Small.” (Section 4.3 Object Detection and Segmentation on COCO) |

## 6. Input and Representation Constraints

- Input resolution is defined as 2D images with variable length: “Given an input of  $H \times W$  resolution” and “process variable-length inputs on the fly.” (Section 3.2 Twins-SVT).
- Window partitioning assumes divisibility: “Without loss of generality, we assume H%m=0 and W%n=0.” (Section 3.2 Twins-SVT).
- Patch embedding and hierarchical output sizes are fixed per stage: “Patch Embedding                 | $P_1 = 4; C_1 = 64$” and “Stage 1 $\frac{H}{4}$   | $\left  \frac{H}{4} \times \frac{W}{4} \right $” (Table 9 – Configuration details of Twins-PCPVT).
- Classification uses fixed 224×224 inputs: “All models are trained and evaluated on 224×224 resolution on ImageNet-1K dataset.” (Table 1).
- Detection/segmentation use resized inputs and specific evaluation resolutions: “randomly resizing the input image so that its shorter side is between 480 and 800 while keeping longer one less than 1333.” (Section 4.3); “FLOPs are tested on 512×512 resolution.” (Table 2); “FLOPs are evaluated on a  $800 \times 600$  image.” (Table 4).

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Sequence length fixed or variable: variable — “process variable-length inputs on the fly.” (Section 3.2 Twins-SVT).
- Attention type(s): global attention in Twins-PCPVT (“The first method is built upon PVT [8] and CPVT [9], which only uses the global attention.” Section 3 Our Method) and local+global in Twins-SVT (“Our proposed SSSA is composed of two types of attention operations—(i) locally-grouped self-attention (LSA), and (ii) global sub-sampled attention (GSA)” Section 3.2 Twins-SVT).
- Windowed attention structure: “the input is spatially grouped into non-overlapped windows and the standard self-attention is computed only within each sub-window.” (Section 1 Introduction); “we first equally divide the 2D feature maps into sub-windows, making self-attention communications only happen within each sub-window.” (Section 3.2 Twins-SVT).
- Cost-management mechanisms: “we propose the spatially separable self-attention (SSSA) to alleviate this challenge” and “Here, we use a single representative to summarize the important information for each of  $m \times n$ sub-windows and the representative is used to communicate with other sub-windows (serving as the key in self-attention), which can dramatically reduce the cost to  $O(mnHWd) = O(\frac{H^2W^2d}{k_1k_2})$ .” (Section 3.2 Twins-SVT).

## 8. Positional Encoding (Critical Section)

- Mechanism used: conditional positional encoding (CPE) via PEG — “Here, we use the conditional position encoding (CPE) proposed in CPVT [9] to replace the absolute PE in PVT. CPE is conditioned on the inputs and can naturally avoid the above issues of the absolute encodings.” (Section 3.1 Twins-PCPVT); “We use the simplest form of PEG, *i.e.*, a 2D depth-wise convolution without batch normalization.” (Section 3.1 Twins-PCPVT).
- Where applied: “The position encoding generator (PEG) [9], which generates the CPE, is placed after the first encoder block of each stage.” and “It is inserted after the first block in each stage.” (Section 3.1 and Section 3.2).
- Variations/experiments: “PVT [8] introduces the pyramid multi-stage design to better tackle dense prediction tasks such as object detection and semantic segmentation. It inherits the absolute positional encoding designed in ViT [1] and DeiT [2].” and “On the contrary, Swin transformer makes use of the relative positional encodings, which bypasses the above issues.” (Section 3.1); “We have also attempted to replace the relative PE with CPE in Swin, which however does not result in noticeable performance gains” (Section 3.1).

## 9. Positional Encoding as a Variable

- Core research variable: Yes — “the less favored performance of PVT is mainly due to the *absolute positional encodings* employed in PVT [8].” (Section 3.1 Twins-PCPVT).
- Multiple positional encodings compared: Yes — “we use the conditional position encoding (CPE) proposed in CPVT [9] to replace the absolute PE in PVT.” and “We have also attempted to replace the relative PE with CPE in Swin” (Section 3.1 Twins-PCPVT).
- PE choice as secondary for Twins-SVT: “The CPVT-based Swin cannot achieve improved performance with both frameworks, which indicates that our performance improvements should be owing to the paradigm of Twins-SVT instead of the positional encodings.” (Section 4.4 Ablation Studies – Positional Encodings).

## 10. Evidence of Constraint Masking

- Model sizes (Params/FLOPs) are reported across variants: “| Twins-SVT-S (ours)           | 24        | 2.9         | 1059                  | 81.7 (+1.8) |  |  |  |  |  |” and “| Twins-SVT-L (ours)           | 99.2      | 15.1        | 288                   | 83.7 (+5.8) |  |  |  |  |  |” (Table 1).
- Dataset size explicitly stated for ADE20K: “This dataset contains 20K images for training and 2K images for validation.” (Section 4.2 Semantic Segmentation on ADE20K).
- Performance gains attributed to positional encoding design for PVT: “the less favored performance of PVT is mainly due to the *absolute positional encodings* employed in PVT [8].” (Section 3.1 Twins-PCPVT).
- Performance gains for Twins-SVT attributed to architecture (not PE): “The CPVT-based Swin cannot achieve improved performance with both frameworks, which indicates that our performance improvements should be owing to the paradigm of Twins-SVT instead of the positional encodings.” (Section 4.4 Ablation Studies – Positional Encodings).
- Training tricks de-emphasized: “Note that we do not utilize extra tricks in [26, 28] to make fair comparisons although it may further improve the” (Section 4.1 Classification on ImageNet-1K).

## 11. Architectural Workarounds

- Windowed local attention to reduce cost: “we first equally divide the 2D feature maps into sub-windows, making self-attention communications only happen within each sub-window.” (Section 3.2 Twins-SVT).
- Global sub-sampled attention for cross-window communication at lower cost: “Here, we use a single representative to summarize the important information for each of  $m \times n$ sub-windows and the representative is used to communicate with other sub-windows (serving as the key in self-attention), which can dramatically reduce the cost to  $O(mnHWd) = O(\frac{H^2W^2d}{k_1k_2})$ .” (Section 3.2 Twins-SVT).
- Interleaving local and global attention (SSSA) for efficiency: “Our proposed SSSA is composed of two types of attention operations—(i) locally-grouped self-attention (LSA), and (ii) global sub-sampled attention (GSA)” (Section 3.2 Twins-SVT).
- Hierarchical pyramid stages with patch embeddings: “PVT [8] introduces the pyramid multi-stage design to better tackle dense prediction tasks such as object detection and semantic segmentation.” (Section 3.1 Twins-PCPVT) and “Stage 1 $\frac{H}{4}$   | $\left  \frac{H}{4} \times \frac{W}{4} \right $” (Table 9).
- Positional encoding generator per stage and classification head adjustment: “The position encoding generator (PEG) [9], which generates the CPE, is placed after the first encoder block of each stage.” and “For image-level classification, following CPVT, we remove the class token and use global average pooling (GAP) at the end of the stage [9].” (Section 3.1 Twins-PCPVT).

## 12. Explicit Limitations and Non-Claims

- Limitations not specified.
- Explicit non-claims not specified.

### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: Single modality (images) with multiple datasets (ImageNet-1K, ADE20K, COCO).
- Task structure: Multiple vision tasks (classification, detection, semantic/instance segmentation) evaluated under separate frameworks.
- Representation rigidity: 2D feature maps with fixed patch sizes/staged resolutions and window divisibility; variable-length inputs allowed.
- Model sharing vs specialization: ImageNet-1k pretraining for ADE20K; detection/segmentation use backbone in RetinaNet/Mask R-CNN.
- Role of positional encoding: Central experimental variable (CPE vs absolute/relative), applied per stage.

### 14. Final Classification

Classification: **Multi-task, single-domain**. The paper evaluates multiple image tasks—“image-level classification as well as dense detection and segmentation.” (Abstract)—with results on ImageNet-1K (Section 4.1), ADE20K semantic segmentation (Section 4.2), and COCO detection/segmentation (Section 4.3), all within the image modality.
