## 1. Basic Metadata

- Title: Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions
- Authors: Wenhai Wang; Enze Xie; Xiang Li; Deng-Ping Fan; Kaitao Song; Ding Liang; Tong Lu; Ping Luo; Ling Shao
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

The paper proposes PVT, a "pure Transformer backbone designed for various pixel-level dense prediction tasks" that "overcomes the difficulties of porting Transformer to various dense prediction tasks" while remaining convolution-free (Introduction, Contributions; Abstract).

## 3. Tasks Evaluated

| Task name | Task type | Dataset(s) used | Domain | Evidence (verbatim quotes with section) |
|---|---|---|---|---|
| Image classification | Classification | ImageNet 2012 (ImageNet) | Not explicitly stated (dataset name only) | "Image classification is the most classical task of imagelevel prediction." (Section 4.1. Image-Level Prediction) <br> "Image classification experiments are performed on the ImageNet 2012 dataset [51]" (Section 5.1. Image Classification) |
| Object detection | Detection | COCO train2017/val2017 | Not explicitly stated (dataset name only) | "Object detection experiments are conducted on the challenging COCO benchmark [40]. All models are trained on COCO train2017 (118k images) and evaluated on val2017 (5k images)." (Section 5.2. Object Detection) <br> "Here, we discuss two typical tasks, namely object detection, and semantic segmentation." (Section 4.2. Pixel-Level Dense Prediction) |
| Instance segmentation | Segmentation | COCO val2017 | Not explicitly stated (dataset name only) | "Similar results are found in instance segmentation experiments based on Mask R-CNN, as shown in Table 4." (Section 5.2. Object Detection) <br> "Object detection and instance segmentation performance on COCO val2017." (Table 4 caption) |
| Semantic segmentation | Segmentation | ADE20K | Not explicitly stated (dataset name only) | "We choose ADE20K [83], a challenging scene parsing dataset, to benchmark the performance of semantic segmentation." (Section 5.3. Semantic Segmentation) <br> "We evaluate our PVT backbones on the basis of Semantic FPN [32]" (Section 5.3. Semantic Segmentation) |

## 4. Domain and Modality Scope

- Modality/domain scope: Single modality (images) with multiple datasets; "given an input image of size  $H \times W \times 3$ " (Section 3.1. Overall Architecture) and "downstream tasks, including image-level prediction as well as pixel-level dense predictions." (Introduction).
- Multiple domains within the same modality vs. multiple modalities: Multiple datasets within the same modality are evaluated; evidence is limited to image inputs and image tasks, e.g., "given an input image of size  $H \times W \times 3$ " (Section 3.1. Overall Architecture).
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
|---|---|---|---|---|
| Image classification | Not stated; task-specific training described | No (trained from scratch) | Yes (classification token + FC head) | "For image classification, we follow ViT [13] and DeiT [63] to append a learnable classification token to the input of the last stage, and then employ a fully connected (FC) layer to conduct classification on top of the token." (Section 4.1) <br> "All models are trained for 300 epochs from scratch on 8 V100 GPUs." (Section 5.1) |
| Object detection | Not stated; ImageNet-pretrained backbone per task | Yes (ImageNet pretrain, then train detection) | Yes (FPN + detector head) | "Like ResNet, we initialize the PVT backbone with the weights pre-trained on ImageNet;" (Section 4.2) <br> "We use the output feature pyramid  $\{F_1, F_2, F_3, F_4\}$  as the input of FPN [38], and then the refined feature maps are fed to the follow-up detection/segmentation head;" (Section 4.2) |
| Instance segmentation | Not stated; ImageNet-pretrained backbone per task | Yes (ImageNet pretrain, then train instance segmentation) | Yes (Mask R-CNN head) | "Similar results are found in instance segmentation experiments based on Mask R-CNN, as shown in Table 4." (Section 5.2) <br> "Like ResNet, we initialize the PVT backbone with the weights pre-trained on ImageNet;" (Section 4.2) |
| Semantic segmentation | Not stated; ImageNet-pretrained backbone per task | Yes (ImageNet pretrain, then train semantic segmentation) | Yes (Semantic FPN head) | "We evaluate our PVT backbones on the basis of Semantic FPN [32]" (Section 5.3) <br> "Like ResNet, we initialize the PVT backbone with the weights pre-trained on ImageNet;" (Section 4.2) |

## 6. Input and Representation Constraints

- 2D image input assumption: "given an input image of size  $H \times W \times 3$ " (Section 3.1. Overall Architecture).
- Fixed patch size at the first stage: "we first divide it into  $\frac{HW}{4^2}$  patches, <sup>2</sup> each of size  $4 \times 4 \times 3$ ." (Section 3.1. Overall Architecture).
- Stage-wise patching and shrinking: "we denote the patch size of the i-th stage as  $P_i$ ... divide the input feature map  $F_{i-1}$  ... into  $\frac{H_{i-1} W_{i-1}}{P_i^2}$  patches" (Section 3.2. Feature Pyramid for Transformer).
- Variable input shapes for detection/segmentation: "Since the input for detection/segmentation can be an arbitrary shape" (Section 4.2. Pixel-Level Dense Prediction).
- Resizing/cropping constraints in evaluation: "a  $224 \times$ 224 patch is cropped to evaluate the classification accuracy." (Section 5.1. Image Classification); "The training image is resized to have a shorter side of 800 pixels, while the longer side does not exceed 1,333 pixels... In the testing phase, the shorter side of the input image is fixed to 800 pixels." (Section 5.2. Object Detection); "We randomly resize and crop the image to  $512 \times 512$  for training, and rescale to have a shorter side of 512 pixels during testing." (Section 5.3. Semantic Segmentation).
- Padding requirements: Padding not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not explicitly stated; sequence length depends on input size and patching, e.g., "divide it into  $\frac{HW}{4^2}$  patches" (Section 3.1. Overall Architecture).
- Fixed or variable sequence length: Variable with input size, e.g., "given an input image of size  $H \times W \times 3$ " (Section 3.1. Overall Architecture).
- Attention type: Hierarchical (pyramid) with spatial-reduction attention; "introducing a progressive shrinking pyramid to reduce the sequence length of Transformer as the network deepens" (Introduction) and "our SRA reduces the spatial scale of K and V before the attention operation" (Section 3.3. Transformer Encoder).
- Mechanisms to manage computational cost: progressive shrinking pyramid and SRA; "uses a progressive shrinking pyramid to reduce the computations of large feature maps" (Abstract) and "adopting a spatial-reduction attention (SRA) layer to further reduce the resource consumption" (Introduction).

## 8. Positional Encoding (Critical Section)

- Mechanism and placement: Position embeddings are added to patch embeddings before the Transformer encoder: "the embedded patches along with a position embedding are passed through a Transformer encoder" (Section 3.1. Overall Architecture).
- Task-specific modification: For detection/segmentation inputs of arbitrary shape, position embeddings are interpolated: "Since the input for detection/segmentation can be an arbitrary shape, the position embeddings pre-trained on ImageNet may no longer be meaningful. Therefore, we perform bilinear interpolation on the pre-trained position embeddings according to the input resolution." (Section 4.2. Pixel-Level Dense Prediction).
- Alternatives or ablations: Not reported.

## 9. Positional Encoding as a Variable

- Positional encoding as a research variable: The paper uses position embeddings as a fixed component and adjusts them for resolution, e.g., "the embedded patches along with a position embedding are passed through a Transformer encoder" (Section 3.1) and "we perform bilinear interpolation on the pre-trained position embeddings according to the input resolution." (Section 4.2).
- Multiple positional encodings compared: Not stated.
- Claim that PE choice is secondary or "not critical": Not stated.

## 10. Evidence of Constraint Masking

- Model size scaling: "we describe a series of PVT models with different scales, namely PVT-Tiny, - Small, -Medium, and -Large, in Table 1, whose parameter numbers are comparable to ResNet18, 50, 101, and 152 respectively." (Section 3.4. Model Details).
- Dataset sizes: "Image classification experiments are performed on the ImageNet 2012 dataset [51], which comprises 1.28 million training images and 50K validation images" (Section 5.1. Image Classification); "All models are trained on COCO train2017 (118k images) and evaluated on val2017 (5k images)." (Section 5.2. Object Detection); "ADE20K contains 150 fine-grained semantic categories, with 20,210, 2,000, and 3,352 images for training, validation, and testing, respectively." (Section 5.3. Semantic Segmentation).
- Architectural hierarchy emphasized: "A Pyramid structure is crucial when applying Transformer to dense prediction tasks." (Section 5.5. Ablation Study).
- Training tricks: "we initialize the PVT backbone with the weights pre-trained on ImageNet" (Section 4.2) and "pre-training weights can also help PVT-based models converge faster and better." (Section 5.5. Ablation Study).
- Model scaling direction: "the deep model (*i.e.*, PVT-Medium) consistently works better than the wide model (*i.e.*, PVT-Small-Wide)" (Section 5.5. Ablation Study).

## 11. Architectural Workarounds

- Progressive shrinking pyramid to reduce sequence length and compute: "introducing a progressive shrinking pyramid to reduce the sequence length of Transformer as the network deepens, significantly reducing the computational cost" (Introduction).
- Spatial-reduction attention (SRA) to reduce attention cost: "our SRA reduces the spatial scale of K and V before the attention operation" (Section 3.3. Transformer Encoder).
- Hierarchical feature pyramid with multi-scale outputs: "The entire model is divided into four stages, each of which is comprised of a patch embedding layer and a  $L_i$ -layer Transformer encoder. Following a pyramid structure, the output resolution of the four stages progressively shrinks from high (4-stride) to low (32-stride)." (Figure 3 caption).
- Task-specific heads via FPN/detectors/segmenters: "We use the output feature pyramid  $\{F_1, F_2, F_3, F_4\}$  as the input of FPN [38], and then the refined feature maps are fed to the follow-up detection/segmentation head;" (Section 4.2).
- Classification token + FC head for classification: "append a learnable classification token to the input of the last stage, and then employ a fully connected (FC) layer to conduct classification on top of the token." (Section 4.1).
- Positional embedding interpolation for variable-resolution inputs: "we perform bilinear interpolation on the pre-trained position embeddings according to the input resolution." (Section 4.2).

## 12. Explicit Limitations and Non-Claims

- Explicit limitations: "there are still some specific modules and operations designed for CNNs and not considered in this work, such as SE [23], SK [36], dilated convolution [74], model pruning [20], and NAS [61]." (Section 6. Conclusions and Future Work).
- Maturity limitation: "the Transformer-based model in computer vision is still in its early stage of development." (Section 6. Conclusions and Future Work).
- Future work scope: "there are many potential technologies and applications (*e.g.*, OCR [68, 66, 69], 3D [28, 11, 27] and medical [15, 16, 29] image analysis) to be explored in the future" (Section 6. Conclusions and Future Work).
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Single image modality with multiple datasets; no cross-domain claims.
> – Task structure: Multiple supervised vision tasks (classification, detection, instance and semantic segmentation) evaluated separately.
> – Representation rigidity: Fixed patch-based 2D inputs with stage-wise patch sizes and task-specific resizing/cropping.
> – Model sharing vs specialization: ImageNet-pretrained backbone fine-tuned per downstream task with task-specific heads; no joint multi-task training stated.
> – Role of positional encoding: Absolute position embeddings added to patch embeddings and interpolated for variable resolutions; not treated as a research variable.

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple tasks: "image classification, object detection, instance and semantic segmentation" (Introduction, Contributions) on ImageNet, COCO, and ADE20K (Sections 5.1-5.3), all within the image modality ("input image of size  $H \times W \times 3$ ", Section 3.1). No cross-domain or cross-modality transfer is claimed.
