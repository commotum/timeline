## 1. Basic Metadata

- Title: "MaxViT: Multi-Axis Vision Transformer" (Title block)
- Authors: "Zhengzhong Tu<sup>1,2</sup>, Hossein Talebi<sup>1</sup>, Han Zhang<sup>1</sup>, Feng Yang<sup>1</sup>, Peyman Milanfar<sup>1</sup>, Alan Bovik<sup>2</sup>, and Yinxiao Li<sup>1</sup>" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper introduces "an efficient and scalable attention model we call multi-axis attention" to enable "global-local spatial interactions on arbitrary input resolutions with only linear complexity" and builds a "simple hierarchical vision backbone, dubbed MaxViT" (Abstract).

---

## 3. Tasks Evaluated

### Task: Image classification (ImageNet-1K)
- Task type: Classification
- Dataset(s) used: ImageNet-1K; ImageNet-21K (pre-training); JFT-300M (pre-training)
- Domain: Images
- Evidence: "We validated the efficacy of our proposed model on various vision tasks: ImageNet classification [48]" (Section 4 Experiments). "ImageNet-1K. We show in Table 2 the performance comparisons on ImageNet-1K classification." (Section 4.1). "Table 3 shows the results of models pre-trained on ImageNet-21K." (Section 4.1). "**JFT-300M.** We also trained our model on a larger-scale proprietary dataset JFT-300M which contains  $\sim \!\! 300$  million weakly labeled images." (Section 4.1).

### Task: Object detection (bounding box detection)
- Task type: Detection
- Dataset(s) used: COCO2017
- Domain: Images
- Evidence: "We validated the efficacy of our proposed model on various vision tasks: ImageNet classification [48], image object detection and instance segmentation [53]" (Section 4 Experiments). "We evaluated the MaxViT architectures on the COCO2017 [53] object bounding box detection and instance segmentation tasks with a two-stage framework [65]." (Section 4.2).

### Task: Instance segmentation
- Task type: Segmentation
- Dataset(s) used: COCO2017
- Domain: Images
- Evidence: "We validated the efficacy of our proposed model on various vision tasks: ImageNet classification [48], image object detection and instance segmentation [53]" (Section 4 Experiments). "We evaluated the MaxViT architectures on the COCO2017 [53] object bounding box detection and instance segmentation tasks" (Section 4.2).

### Task: Image aesthetics / quality assessment
- Task type: Other (aesthetic/quality assessment)
- Dataset(s) used: AVA benchmark
- Domain: Images
- Evidence: "We validated the efficacy of our proposed model on various vision tasks: ImageNet classification [48], image object detection and instance segmentation [53], image aesthetics/quality assessment [61]" (Section 4 Experiments). "We train and evaluate the MaxViT model on the AVA benchmark [61] which contains 255K images with aesthetics scores rated by amateur photographers." (Section 4.3).

### Task: Unconditional image generation
- Task type: Generation
- Dataset(s) used: ImageNet-1K
- Domain: Images
- Evidence: "We evaluate the generative ability of MaxViT blocks to generate images of 128x128 resolution on ImageNet-1K. We choose the unconditional image generation to focus on the performance of different generators in GANs." (Section 4.4).

---

## 4. Domain and Modality Scope

- Evaluation scope: Multiple domains within the same modality (vision/image tasks). Evidence: "We validated the efficacy of our proposed model on various vision tasks: ImageNet classification [48], image object detection and instance segmentation [53], image aesthetics/quality assessment [61], and unconditional image generation [26]." (Section 4 Experiments).
- Domain generalization or cross-domain transfer: Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image classification (ImageNet-1K) | Yes (ImageNet-1K weights reused for downstream tasks) | Yes (higher-resolution fine-tuning reported) | Yes | "For all the compared models, the backbones are first pretrained using ImageNet-1K." (Section 4.2); "When fine-tuned at higher resolutions (384/512), MaxViT continues to deliver high performance" (Section 4.1); "Instead of using the [cls] token [22], we simply apply global average pooling to the output of the last stage (S4) to obtain the feature representation, followed by the final classification head." (Appendix A.1). |
| Object detection (COCO2017) | Yes (ImageNet-1K pretrained backbone) | Yes | Yes | "For all the compared models, the backbones are first pretrained using ImageNet-1K. The pretrained models are then used to finetune on the detection and segmentation tasks." (Section 4.2); "On the object detection task, a feature-pyramid architecture [52] was employed... Then the generated feature maps are fed into the detection head." (Section 4.2). |
| Instance segmentation (COCO2017) | Yes (ImageNet-1K pretrained backbone) | Yes | Yes | "For all the compared models, the backbones are first pretrained using ImageNet-1K. The pretrained models are then used to finetune on the detection and segmentation tasks." (Section 4.2); "In the instance segmentation task, a well-known Cascade Mask-RCNN framework [28] was employed." (Section 4.2). |
| Image aesthetics / quality assessment (AVA) | Yes (ImageNet-1K pretrained weights) | Yes | Yes | "We initialized the model with ImageNet-1K  $224 \times 224$  pre-trained weights." (Section B.3); "We remove the classification head used in MaxViT, and instead append a fully-connected layer with 10 neurons followed by softmax." (Appendix A.3). |
| Unconditional image generation (ImageNet-1K) | Not specified | Not specified | N/A (GAN generator/discriminator) | "We evaluate the generative ability of MaxViT blocks to generate images of 128x128 resolution on ImageNet-1K." (Section 4.4); "MaxViT-GAN first takes a latent code  $z \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$  as input, then progressively generates an image of target resolution" (Appendix A.4); "We use a ResNet-based discriminator following [42]." (Section B.4). |

---

## 6. Input and Representation Constraints

- Variable input resolution supported: "global-local spatial interactions on arbitrary input resolutions" (Abstract).
- 2D spatial representation assumption: "Let  $X \in \mathbb{R}^{H \times W \times C}$  be an input feature map." (Section 3.2). "# input: features (b, h, w, c). Assume h == w; x/output: features (b, h, w, c)." (Algo. 1).
- Fixed window size for block attention: "partitioning into non-overlapping windows, each of size  $P \times P$." (Section 3.2).
- Fixed grid size for grid attention: "we grid the tensor into the shape  $(G \times G, \frac{H}{G} \times \frac{W}{G}, C)$  using a fixed  $G \times G$  uniform grid" (Section 3.2).
- Specific window/grid hyperparameters: "we use P = G = 7 following Swin [56]" (Section 3.2). "# p/g: block/grid size. Use 7 by default." (Algo. 1).
- Explicit statement of no padding/masking in attention: "it enjoys global interaction capability without requiring masking, padding, or cyclic-shifting" (Section 3.2).
- Experiment-specific resizing constraints: "Under the basic 224×224 setting" (Section 4.1); "For both tasks, the input images are resized to  $896 \times 896$." (Section B.2); "We train MaxViT for three different input resolutions:  $224 \times 224$ ,  $384 \times 384$  and  $512 \times 512$." (Section B.3); "generate images of 128x128 resolution on ImageNet-1K" (Section 4.4).
- Fixed patch size: Not specified.
- Fixed number of tokens: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified. Largest evaluated input size is " $896 \times 896$ " for detection/segmentation (Section B.2).
- Fixed or variable sequence length: Variable. "global-local spatial interactions on arbitrary input resolutions" (Abstract).
- Attention type: Blocked local + grid/global, hierarchical. "multi-axis attention, which consists of two aspects: blocked local and dilated global attention" (Abstract). "We will use this **block attention** to conduct local interactions." and "Employing self-attention on the decomposed grid axis... corresponds to dilated, global spatial mixing of tokens." (Section 3.2). "a simple hierarchical vision backbone... by simply repeating the basic building block over multiple stages." (Abstract).
- Computational cost management: "decomposing the fully dense attention mechanisms into two sparse forms – window attention and grid attention – which reduces the quadratic complexity of vanilla attention to linear" (Section 3 Method). "both having only linear complexity with respect to spatial size or sequence length." (Section 3.2).

---

## 8. Positional Encoding (Critical Section)

- Mechanism: Relative position bias in attention; implicit positional encoding via depthwise conv.
  - "Relative self-attention... introducing a relative learned bias added to the attention weights" (Section 3.1).
  - "RelAttention(Q, K, V) = softmax(QK^{T}/\sqrt{d} + B)V" (Appendix A.1).
  - "depthwise convolutions can be regarded as conditional position encoding (CPE) [17], making our model free of explicit positional encoding layers." (Section 3.2).
- Where applied: Attention bias inside attention weights (Appendix A.1), plus implicit CPE via MBConv (Section 3.2).
- Fixed vs modified: Default relative attention everywhere, with interpolation when changing resolution.
  - "In our model, all the attention operators use this relative attention defined in Eq. 3 by default." (Appendix A.1).
  - "when fine-tuned at a higher resolution e.g.,  $H' \times W'$ , we use bilinear interpolation to map the relative positional bias" (Appendix A.1).
- Ablated or compared against alternatives: Not specified.

---

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption: Fixed architectural assumption. Evidence: "In this work, we mainly adopt the pre-normalized relative self-attention defined in [19] as the key operator in MaxViT." (Section 3.1). "In our model, all the attention operators use this relative attention defined in Eq. 3 by default." (Appendix A.1).
- Multiple positional encodings compared: Not specified.
- Claim that PE choice is not critical or secondary: Not stated.

---

## 10. Evidence of Constraint Masking

- Model size scaling: "MaxViT-XL achieves a high accuracy of 89.53% with 475 million parameters" (Section 4.1).
- Dataset size scaling: "JFT-300M... contains  $\sim \!\! 300$  million weakly labeled images." (Section 4.1).
- Scaling model size as driver: "MaxViT scales much better than SOTA vision Transformers on the ImageNet-1K trained model scale." (Section 4.1).
- Scaling data as driver: "our model is also scalable to massive scale training data – MaxViT-XL achieves a high accuracy of 89.53%" (Section 4.1).
- Architectural hierarchy emphasized: "a simple hierarchical vision backbone, dubbed MaxViT, by simply repeating the basic building block over multiple stages." (Abstract).
- Training tricks: "Notably, we do not employ extra GAN training tricks such as pixel norm, noise injection, progressive growing, etc." (Section B.4).

---

## 11. Architectural Workarounds

- Windowed (block) attention to localize interactions: "partitioning into non-overlapping windows, each of size  $P \times P$. Applying self-attention on the local spatial dimension" (Section 3.2).
- Grid attention for sparse global mixing: "Employing self-attention on the decomposed grid axis... corresponds to dilated, global spatial mixing of tokens." (Section 3.2).
- Linear-complexity decomposition: "decomposing the fully dense attention mechanisms into two sparse forms – window attention and grid attention – which reduces the quadratic complexity of vanilla attention to linear" (Section 3 Method).
- Hierarchical stages / repeated blocks: "a simple hierarchical vision backbone... by simply repeating the basic building block over multiple stages." (Abstract).
- MBConv before attention: "We also add a MB-Conv block [35] with squeeze-and-excitation (SE) module [36] prior to the multiaxis attention" (Section 3.2).
- Downsampling within stages: "We apply downsampling in the Depthwise Conv3x3 layer of the first MBConv block in each stage." (Section 3.3).

---

## 12. Explicit Limitations and Non-Claims

- Resource limitations / future work: "Due to resource limitations, we leave experiments on billion-parameter-scale models on planet-scale datasets (e.g., JFT-3B [102]) as future work." (Section 4.1).
- No system-level tricks for detection: "For fair comparison, we follow the original implementation without adopting any system-level strategies to further boost the final performance, such as the HTC framework [7], instaboost [25], etc. used in Swin [56]." (Appendix A.2).
- No extra GAN training tricks: "Notably, we do not employ extra GAN training tricks such as pixel norm, noise injection, progressive growing, etc." (Section B.4).

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Multiple image-based vision tasks/datasets within a single modality (images), with no cross-domain transfer claim.
> - Task structure: Separate evaluations for classification, detection/segmentation, aesthetics assessment, and image generation rather than joint multi-task training.
> - Representation rigidity: 2D HxW feature maps with fixed window/grid sizes (P=G=7) and fixed per-task input resolutions, though the model claims arbitrary input resolutions.
> - Model sharing vs specialization: ImageNet-1K pretraining reused for downstream tasks with task-specific heads (detection head, AVA FC head), while GAN uses a separate generator/discriminator setup.
> - Role of positional encoding: Relative position bias in attention with implicit CPE via depthwise conv; fixed by default with interpolation for resolution changes.

---

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple tasks: "ImageNet classification [48], image object detection and instance segmentation [53], image aesthetics/quality assessment [61], and unconditional image generation [26]" (Section 4 Experiments). All evaluations are image-based (ImageNet-1K, COCO2017, AVA, and ImageNet-1K generation) with no explicit cross-domain transfer claim, so the setup is multi-task within a single modality/domain (Sections 4.1-4.4).
