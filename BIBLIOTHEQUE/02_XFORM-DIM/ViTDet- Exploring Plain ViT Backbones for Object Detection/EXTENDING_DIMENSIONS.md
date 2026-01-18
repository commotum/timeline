## 1. Basic Metadata

- Title: "Exploring Plain Vision Transformer Backbones for Object Detection" (Title line)
- Authors: "Yanghao Li Hanzi Mao Ross Girshick<sup>†</sup> Kaiming He<sup>†</sup>" (Title line)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper's primary contribution is to "explore the plain, non-hierarchical Vision Transformer (ViT) as a backbone network for object detection" and show that "with minimal adaptations for fine-tuning, our plain-backbone detector can achieve competitive results" (Abstract).

---

## 3. Tasks Evaluated

- **Task 1 (Bounding-box object detection):** Task type: Detection; Dataset(s): COCO, LVIS; Domain: Not explicitly stated (COCO/LVIS image datasets); Evidence: "We report results on bounding-box object detection (AP<sup>box</sup>) and instance segmentation (AP<sup>mask</sup>)." (Section 4.1) and "We perform ablation experiments on the COCO dataset [39]." (Section 4.1) and "We further report system-level comparisons on the LVIS dataset [23]." (Section 4.3)
- **Task 2 (Instance segmentation):** Task type: Segmentation; Dataset(s): COCO, LVIS; Domain: Not explicitly stated (COCO/LVIS image datasets); Evidence: "We report results on bounding-box object detection (AP<sup>box</sup>) and instance segmentation (AP<sup>mask</sup>)." (Section 4.1) and "We further report system-level comparisons on the LVIS dataset [23]." (Section 4.3) and "LVIS contains ~2M high-quality instance segmentation annotations for 1203 classes that exhibit a natural, long-tailed object distribution." (Section 4.3)

---

## 4. Domain and Modality Scope

- Single domain? Not explicitly stated; evaluation is on COCO and LVIS image datasets: "We perform ablation experiments on the COCO dataset [39]." (Section 4.1) and "We further report system-level comparisons on the LVIS dataset [23]." (Section 4.3)
- Multiple domains within the same modality? Not explicitly stated; the paper reports results on COCO and LVIS: "We perform ablation experiments on the COCO dataset [39]." (Section 4.1) and "We further report system-level comparisons on the LVIS dataset [23]." (Section 4.3)
- Multiple modalities? Not stated.
- Domain generalization or cross-domain transfer? Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Bounding-box object detection | Not explicitly stated. | Yes; fine-tuned for detection. | Not explicitly stated (Mask R-CNN/Cascade Mask R-CNN heads used). | "This design enables the original ViT architecture to be fine-tuned for object detection without needing to redesign a hierarchical backbone for pre-training." (Abstract) and "Our detector heads follow Mask R-CNN [25] or Cascade Mask R-CNN [4]." (Implementation) and "We report results on bounding-box object detection (AP<sup>box</sup>) and instance segmentation (AP<sup>mask</sup>)." (Section 4.1) |
| Instance segmentation | Not explicitly stated. | Yes; fine-tuned for detection/segmentation. | Not explicitly stated (Mask R-CNN/Cascade Mask R-CNN heads used). | "We report results on bounding-box object detection (AP<sup>box</sup>) and instance segmentation (AP<sup>mask</sup>)." (Section 4.1) and "Our detector heads follow Mask R-CNN [25] or Cascade Mask R-CNN [4]." (Implementation) and "This design enables the original ViT architecture to be fine-tuned for object detection without needing to redesign a hierarchical backbone for pre-training." (Abstract) |

---

## 6. Input and Representation Constraints

- Input resolution: "The input image is 1024 × 1024, augmented with large-scale jittering [19] during training." (Implementation) and "increase the input size (from 1024 to 1280) following [36,41]." (Section 4.3)
- Fixed patch size and stride: "We set the patch size as 16 and thus the feature map scale is 1/16, i.e., stride = 16." (Implementation) and "We use a patch size of 16 for all ViT backbones." (Appendix A.2)
- Patch-embedding resizing (representation adjustment): "As ViT-H in [14] by default has a patch size of 14, after pre-training we interpolate the patch embedding filters from 14 × 14 × 3 to 16 × 16 × 3." (Appendix A.2)
- Fixed number of tokens: Not specified.
- Fixed dimensionality (e.g., strictly 2D) and padding/resizing requirements: Not specified beyond the patch-embedding interpolation above.

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Fixed vs. variable sequence length: Not explicitly stated; input sizes are set per experiment (e.g., "increase the input size (from 1024 to 1280)" in Section 4.3).
- Attention type: Windowed with occasional global or convolutional propagation: "we divide it into regular non-overlapping windows. Self-attention is computed within each window. This is referred to as \"restricted\" self-attention" (Section 3) and "Unlike Swin, we do *not* \"shift\" [42] the windows across layers." (Section 3) and "We perform global self-attention in the last block of each subset." (Section 3)
- Mechanisms to manage computational cost: "computing global self-attention throughout the backbone is prohibitive in memory and is slow" and "To efficiently extract features from high-resolution images, our detector uses simple non-overlapping window attention... A small number of cross-window blocks (e.g., 4)... are used to propagate information." (Section 3)

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: The paper mentions "positional embedding [54]" for location encoding (type not specified) and also uses "relative position biases [46]" in some experiments (Sections 3 and 4.2).
- Where applied: "we also adopt relative position biases in our ViT backbones as per [34], but *only* during fine-tuning, not affecting pre-training." (Section 4.2)
- Fixed vs. modified/ablated: "Note that our ablations in Sec. 4.1 are *without* relative position biases." (Section 4.2)

---

## 9. Positional Encoding as a Variable

- Core research variable vs. fixed assumption: Not presented as a core variable; relative position biases are added "for a fairer comparison" (Section 4.2).
- Multiple positional encodings compared? The paper uses settings with and without relative position biases: "we also adopt relative position biases... but *only* during fine-tuning" and "our ablations in Sec. 4.1 are *without* relative position biases." (Section 4.2)
- Claim that PE choice is "not critical" or secondary? Not claimed.

---

## 10. Evidence of Constraint Masking

- Model sizes and scaling: "We use the vanilla ViT-B, ViT-L, ViT-H [14] as the pretraining backbones." (Implementation) and "The gains are more prominent for larger model sizes." (Abstract)
- Dataset scale (pre-training): "We report 61.3 AP<sup>box</sup> on the COCO dataset [39] with a plain ViT-Huge backbone, using only ImageNet-1K pre-training with no labels." (Abstract)
- Dataset scale (downstream): "LVIS contains ~2M high-quality instance segmentation annotations for 1203 classes that exhibit a natural, long-tailed object distribution." (Section 4.3)
- Performance gains attributed to training strategy (MAE): "MAE [24] pre-training on IN-1K (without labels) shows massive gains, increasing AP<sup>box</sup> by 3.1 for ViT-B and 4.6 for ViT-L." (Section 4.1)
- Architectural contributions vs. hierarchy: "our detector builds a simple feature pyramid from only the last feature map of a plain ViT backbone" and "window attention is sufficient as long as information is well propagated across windows in a small number of layers." (Introduction)

---

## 11. Architectural Workarounds

- Simple feature pyramid from a single-scale map to supply multi-scale features: "we simply use only the *last* feature map from the backbone... apply a set of convolutions or deconvolutions in parallel to produce multi-scale feature maps... We refer to this as a \"simple feature pyramid\"." (Section 3)
- Windowed attention to handle high-resolution inputs: "To efficiently extract features from high-resolution images, our detector uses simple non-overlapping window attention (without \"shifting\", unlike [42])." (Introduction)
- Cross-window propagation blocks to spread information: "A small number of cross-window blocks (e.g., 4), which could be global attention [54] or convolutions, are used to propagate information." (Introduction)
- Fine-tuning-only adaptations to avoid changing pre-training: "These adaptations are made only during fine-tuning and do not alter pre-training." (Introduction)

---

## 12. Explicit Limitations and Non-Claims

- "In this work, we do *not* aim to develop new components; instead, we make minimal adaptations that are sufficient to overcome the aforementioned challenges." (Introduction)
- "Changing the stride affects the scale distribution and presents a different accuracy shift for objects of different scales. This topic is beyond the scope of this study." (Footnote 7)
- "This is largely beyond the scope of this paper, as it involves finding good training recipes for hierarchical backbones with MAE." (Section 4.2)
- "A special issue in LVIS is on the long-tailed distribution, which is beyond the scope of our study." (Section 4.3)
- "Exploring even fewer inductive biases in the detection heads is an open and interesting direction for future work." (Discussion)
- "In our study, we focus on leveraging pre-trained plain backbones and we do not constrain the detector neck/head design." (Section 2)

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: Evaluations are on COCO and LVIS datasets ("We perform ablation experiments on the COCO dataset [39]." (Section 4.1) and "We further report system-level comparisons on the LVIS dataset [23]." (Section 4.3)).
> – Task structure: The evaluated tasks are "bounding-box object detection (AP<sup>box</sup>) and instance segmentation (AP<sup>mask</sup>)." (Section 4.1).
> – Representation rigidity: Fixed patch size/stride and set input sizes ("We set the patch size as 16 and thus the feature map scale is 1/16, i.e., stride = 16." (Implementation) and "increase the input size (from 1024 to 1280)" (Section 4.3)).
> – Model sharing vs specialization: Pretrained ViT backbones are fine-tuned with detector heads ("This design enables the original ViT architecture to be fine-tuned for object detection..." (Abstract) and "Our detector heads follow Mask R-CNN [25] or Cascade Mask R-CNN [4]." (Implementation)).
> – Role of positional encoding: Location encoding is via "positional embedding [54]" and, for some comparisons, "relative position biases [46]" added "only during fine-tuning." (Sections 3 and 4.2).

---

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates two tasks—"bounding-box object detection (AP<sup>box</sup>) and instance segmentation (AP<sup>mask</sup>)" (Section 4.1)—using the same visual datasets. Evaluations are on COCO and LVIS ("We perform ablation experiments on the COCO dataset [39]." (Section 4.1); "We further report system-level comparisons on the LVIS dataset [23]." (Section 4.3)) with no multi-modality or cross-domain claims.
