## 1. Basic Metadata

- Title: "Rotary Position Embedding for Vision Transformer" (Title block)
- Authors: "Byeongho Heo"; "Song Park"; "Dongyoon Han"; "Sangdoo Yun" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper provides "a comprehensive analysis of RoPE when applied to ViTs" and states, "we propose to use mixed axis frequencies for 2D RoPE, named RoPE-Mixed." (Abstract; 1 Introduction)

## 3. Tasks Evaluated

- Task name: Multi-resolution classification
  - Task type: Classification
  - Dataset(s) used: ImageNet-1k
  - Domain: Images (vision)
  - Quotes: "multi-resolution classification (§4.1) on ImageNet-1k [4]" (4 Experiments); "We train ViTs and Swin Transformers on ImageNet-1k [4] training set" (4.1 Multi-resolution classification); "2D RoPE for input images" (3.2 RoPE for 2D images)

- Task name: Object detection
  - Task type: Detection
  - Dataset(s) used: MS-COCO
  - Domain: Images (vision)
  - Quotes: "object detection (§4.2) on MS-COCO [16]" (4 Experiments); "We verify 2D RoPE in object detection on MS-COCO [16]." (4.2 Object detection); "2D RoPE for input images" (3.2 RoPE for 2D images)

- Task name: Semantic segmentation
  - Task type: Segmentation
  - Dataset(s) used: ADE20k
  - Domain: Images (vision)
  - Quotes: "semantic segmentation (§4.3) on ADE20k [40,41]." (4 Experiments); "We train 2D RoPE ViT and Swin for semantic segmentation on ADE20k [40, 41]." (4.3 Semantic segmentation); "2D RoPE for input images" (3.2 RoPE for 2D images)

## 4. Domain and Modality Scope

- Evaluation performed on: Multiple domains within the same modality (vision/images). Evidence: "image recognition, including multi-resolution classification (§4.1) on ImageNet-1k [4], object detection (§4.2) on MS-COCO [16], and semantic segmentation (§4.3) on ADE20k [40,41]." (4 Experiments); "2D RoPE for input images" (3.2 RoPE for 2D images)
- Does the paper claim domain generalization or cross-domain transfer?: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Multi-resolution classification (ImageNet-1k) | Yes; ImageNet-1k weights reused as pre-trained backbones for downstream tasks | No; this is the pretraining step | Not specified | "We train ViTs and Swin Transformers on ImageNet-1k [4] training set" (4.1 Multi-resolution classification); "We use ImageNet-1k weights from §4.1 for pre-trained weights" (4.2 Object detection) |
| Object detection (MS-COCO) | Yes; ImageNet-1k pre-trained backbone reused | Yes; detection training | Yes; DINO detector with backbone | "We use ImageNet-1k weights from §4.1 for pre-trained weights" (4.2 Object detection); "DINO [39] detector is trained using ViT and Swin as backbone network." (4.2 Object detection) |
| Semantic segmentation (ADE20k) | Yes; ImageNet-1k pre-trained backbone reused | Yes; segmentation training | Yes; UperNet/Mask2Former head | "ImageNet-1k pretrained weights from §4.1 are used for pre-trained weights." (4.3 Semantic segmentation); "For ViT, we use UperNet [37]" and "For Swin, Mask2Former [2] for segmentation is used with the Swin." (4.3 Semantic segmentation) |

## 6. Input and Representation Constraints

- Input tokenization: "The transformer treats input data as a sequence of tokens." (1 Introduction)
- Patch size: "patchification layer computes tokens from  $16 \times 16$  or  $32 \times 32$  patch images." (3.1 Preliminary: Introducing Position Embeddings)
- 2D positional grid: "we need to change the 1D token index n in RoPE to a 2D token position  $\mathbf{p}_n = (p_n^x, p_n^y)$  where  $p_n^x \in \{0, 1, ..., W\}, p_n^y \in \{0, 1, ..., H\}$  for token width W and height H." (3.2 RoPE for 2D images)
- Fixed training resolution: "we use the ImageNet-1k standard image resolution  $224 \times 224$  for training." (4.1 Multi-resolution classification)
- Variable inference resolution: "We report the accuracy on the ImageNet-1k validation set as varying image sizes." (4.1 Multi-resolution classification)
- Segmentation input resolution: "the ViT-UperNet setting uses 512 × 512 images for inputs." (4.3 Semantic segmentation)
- Fixed number of tokens, padding, resizing: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Fixed or variable sequence length: Variable, as evaluation uses "varying image sizes." (4.1 Multi-resolution classification)
- Attention type: Windowed and hierarchical attention for Swin ("window size of the window attention" and pooling in Swin) and mixed window/global for DINO-ViTDet ("window-block attention" with "global attention" layers). (4.1 Multi-resolution classification; 2 Related Works; 4.2 Object detection)
- Computational cost mechanisms: "we change the window size of the window attention" (Fig. 5 caption, 4.1 Multi-resolution classification); "Swin Transformer [17] increase the spatial length of tokens at early layers using pooling." (2 Related Works); "DINO-ViTDet uses ViT backbone with window-block attention, but still, a few layers remain as global attention." (4.2 Object detection); "The rotation matrix in Eq. 12 and 14 is pre-computed before inference." (3.3 Discussion)

## 8. Positional Encoding (Critical Section)

- Mechanism used: "There are two primary methods in position embedding for Vision Transformers: Absolute Positional Embedding (APE) [5,6] and Relative Position Bias (RPB) [17,23,27]." (1 Introduction); "Rotary Position Embedding (RoPE) [29] was introduced to apply to key and query in self-attention layers as channel-wise multiplications" (3 Method); "we propose to use mixed axis frequencies for 2D RoPE, named RoPE-Mixed." (1 Introduction)
- Where applied: "RoPE [29] was introduced to apply to key and query in self-attention layers as channel-wise multiplications, which is distinct from conventional position embeddings - APE is added to the stem layer; RPB is added to an attention matrix." (3 Method)
- Fixed vs modified across experiments: "We compare the conventional position embeddings (APE, RPB) with two variants of 2D RoPE RoPE-Axial (Eq. 12) and RoPE-Mixed (Eq. 14)." (4 Experiments); "When applying RoPE to ViT, we remove APE from ViT by default." and "we replace RPB with 2D RoPE for comparison." (4.1 Multi-resolution classification); "We measure the performance of RoPE-Mixed when it is used with APE." and "We also measure performance when RoPE-Mixed is used together with RPB." (4.1 Multi-resolution classification)

## 9. Positional Encoding as a Variable

- Core research variable?: Yes. "This paper aims to improve position embedding for vision transformers by applying an extended Rotary Position Embedding (RoPE) [29]." (1 Introduction)
- Multiple positional encodings compared?: Yes. "We compare the conventional position embeddings (APE, RPB) with two variants of 2D RoPE RoPE-Axial (Eq. 12) and RoPE-Mixed (Eq. 14)." (4 Experiments)
- PE choice claimed "not critical" or secondary?: Not stated.

## 10. Evidence of Constraint Masking

- Model size(s) referenced: "We apply 2D RoPE to ViT-S, ViT-B, and ViT-L." and "We train Swin-T, Swin-S, and Swin-B" (4.1 Multi-resolution classification); "accounts for only 0.01% of ViT-B's 17.6G FLOPs." (3.3 Discussion)
- Dataset size(s): Dataset sizes not specified; datasets named include "ImageNet-1k," "MS-COCO," and "ADE20k." (4 Experiments)
- Performance gains attributed to PE changes (not scaling): "DINO-ViTDet achieves AP improvement of more than +1.0pp by changing positional embedding to RoPE." (4.2 Object detection); "Mixed+APE achieves +2.3 and +2.5 mIoU improvement with only position embedding changes." (4.3 Semantic segmentation)
- Training recipes referenced (held constant in experiments): "DeiT-III [32]'s 400 epochs training recipe" and "Swin Transformer 300epochs training recipe" (4.1 Multi-resolution classification)

## 11. Architectural Workarounds

- Hierarchical stages/pooling: "Swin Transformer [17] increase the spatial length of tokens at early layers using pooling." (2 Related Works)
- Windowed attention for efficiency: "For multi-resolution inference, we change the window size of the window attention." (Fig. 5 caption, 4.1 Multi-resolution classification)
- Window-block attention with some global layers: "DINO-ViTDet uses ViT backbone with window-block attention, but still, a few layers remain as global attention." (4.2 Object detection)
- Fixed grid patchification: "patchification layer computes tokens from  $16 \times 16$  or  $32 \times 32$  patch images." (3.1 Preliminary: Introducing Position Embeddings)
- Precomputation for RoPE efficiency: "The rotation matrix in Eq. 12 and 14 is pre-computed before inference." (3.3 Discussion)

## 12. Explicit Limitations and Non-Claims

No explicit limitations, future work, or non-claims are stated in the provided text.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: The work targets vision inputs such as "input images" in the "vision domain." (3.2 RoPE for 2D images)
> – Task structure: Evaluated on "multi-resolution classification (§4.1) on ImageNet-1k [4], object detection (§4.2) on MS-COCO [16], and semantic segmentation (§4.3) on ADE20k [40,41]." (4 Experiments)
> – Representation rigidity: Uses patchified tokens from "$16 \times 16$  or  $32 \times 32$  patch images" with 2D grid positions $(p_n^x, p_n^y)$, and training at "$224 \times 224$." (3.1; 3.2; 4.1)
> – Model sharing vs specialization: "ImageNet-1k weights from §4.1" are reused as "pre-trained weights" for downstream tasks. (4.2; 4.3)
> – Role of positional encoding: Central variable comparing "APE, RPB" with "RoPE-Axial" and "RoPE-Mixed." (4 Experiments)

### 14. Final Classification

Classification: **Multi-task, single-domain.** The evaluation spans multiple vision tasks—"multi-resolution classification (§4.1) on ImageNet-1k [4], object detection (§4.2) on MS-COCO [16], and semantic segmentation (§4.3) on ADE20k [40,41]"—all on "input images" in the vision modality. (4 Experiments; 3.2 RoPE for 2D images) The models reuse ImageNet-1k pretraining for downstream detection and segmentation, indicating separate fine-tuning rather than joint multi-domain training. (4.2 Object detection; 4.3 Semantic segmentation)
