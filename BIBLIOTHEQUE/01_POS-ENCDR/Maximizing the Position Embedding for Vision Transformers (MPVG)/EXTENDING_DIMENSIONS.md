## 1. Basic Metadata

- Title: Maximizing the Position Embedding for Vision Transformers with Global Average Pooling. Evidence: "Maximizing the Position Embedding for Vision Transformers with Global Average Pooling" (Title block).
- Authors: Wonjun Lee; Bumsub Ham; Suhyun Kim. Evidence: "Wonjun Lee<sup>1,2</sup>, Bumsub Ham<sup>1</sup>, Suhyun Kim<sup>2\*</sup>" (Title block).
- Year: 2025. Evidence: "Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved." (Introduction).
- Venue: Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper proposes MPVG to "maximize the effectiveness of PE in the GAP approach" and address the "conflicting result where performance decreased when the GAP and Layer-wise structure were applied together." (Abstract; Introduction)

---

## 3. Tasks Evaluated

| Task name | Task type | Dataset(s) used | Domain | Evidence |
| --- | --- | --- | --- | --- |
| Image classification | Classification | ImageNet-1K; CIFAR-100 | Images / computer vision | "We evaluate the performance of our methods on ImageNet-1K (Deng et al. 2009) and CIFAR-100 (Krizhevsky, Hinton et al. 2009)." (Experiment - Image Classification). "computer vision tasks such as image classification, object detection, and semantic segmentation." (Related Work - Vision Transformers) |
| Object detection | Detection | COCO 2017 | Images / computer vision | "On object detection, we evaluate our methods on COCO 2017 (Lin et al. 2014)." (Object Detection). "computer vision tasks such as image classification, object detection, and semantic segmentation." (Related Work - Vision Transformers) |
| Semantic segmentation | Segmentation | ADE20K | Images / computer vision | "On semantic segmentation, we evaluate our methods on ADE20K (Zhou et al. 2019)." (Semantic Segmentation). "computer vision tasks such as image classification, object detection, and semantic segmentation." (Related Work - Vision Transformers) |

---

## 4. Domain and Modality Scope

- Evaluation scope: Single modality (vision/images) across computer vision datasets. Evidence: "vision transformers have become essential architecture in the field of computer vision" (Introduction); "We evaluate the performance of our methods on ImageNet-1K ... and CIFAR-100" (Experiment - Image Classification); "On object detection, we evaluate our methods on COCO 2017" (Object Detection); "On semantic segmentation, we evaluate our methods on ADE20K" (Semantic Segmentation).
- Multiple domains or modalities: Multiple datasets within the same modality; no explicit multiple modalities are described. Evidence: "ImageNet-1K ... CIFAR-100" (Experiment - Image Classification); "COCO 2017" (Object Detection); "ADE20K" (Semantic Segmentation).
- Domain generalization or cross-domain transfer: Not claimed. The paper notes dataset transfer within vision: "we transfer our pretrained T2T-ViT to downstream datasets such as CIFAR-100 and finetune the pretrained T2T-ViT-7" (Experiment - Image Classification).

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image classification (ImageNet-1K) | No explicit sharing across tasks; trained per dataset | Not specified | MLP head mentioned for classification | "We evaluate the performance of our methods on ImageNet-1K" (Experiment - Image Classification). "All vision transformers are trained on 224×224 resolution images for 300 epochs" (Experiment - Image Classification). "the output of this token is then used to make class predictions via Multi-Layer Perceptron (MLP)" (Introduction) |
| Image classification (CIFAR-100) | Pretrained T2T-ViT reused for CIFAR-100; otherwise trained per dataset | Yes, for T2T-ViT-7 | MLP head mentioned for classification | "we transfer our pretrained T2T-ViT to downstream datasets such as CIFAR-100 and finetune the pretrained T2T-ViT-7" (Experiment - Image Classification). "ViT-Lite was trained for 310 epochs on 32×32 resolution images" (Experiment - Image Classification). "the output of this token is then used to make class predictions via Multi-Layer Perceptron (MLP)" (Introduction) |
| Object detection (COCO 2017) | ImageNet-1K pretrained DeiT-Ti backbone used | Yes (pretrained backbone) | Yes (Mask R-CNN) | "For comparison, DeiT-Ti model pretrained on ImageNet-1K with each method is used." (Table 3). "we select the ViT-Adapter-Ti (Chen et al. 2022) model based on Mask R-CNN (He et al. 2017)" (Object Detection) |
| Semantic segmentation (ADE20K) | ImageNet-1K pretrained DeiT-Ti backbone used | Yes (pretrained backbone) | Yes (UperNet) | "For comparison, DeiT-Ti model pretrained on ImageNet-1K with each method is used." (Table 4). "We select the ViT-Adapter-Ti (Chen et al. 2022) model based on UperNet (Xiao et al. 2018)" (Semantic Segmentation) |

---

## 6. Input and Representation Constraints

- Patch-based tokenization and token count: "N represents the number of patches, calculated as HW/P^2, where H and W are the height and width of the image, and P × P is the resolution of each patch." (Preliminary: Absolute Position Embedding).
- Example token count reported for analysis: "the y-axis represents the number of tokens (196)." (Figure 2).
- Fixed training resolutions for classification: "All vision transformers are trained on 224×224 resolution images for 300 epochs" and "ViT-Lite was trained for 310 epochs on 32×32 resolution images" (Experiment - Image Classification).
- Fixed crop sizes for dense tasks: "Crop Size 1024" (Table 8) and "Crop Size 512" (Table 9).
- Class token removed under GAP: "We remove the class token as we adapt the Global Average Pooling (GAP) method." (Maximizing the Position Embedding with GAP).
- Patch size value not specified beyond P × P. Evidence: "P × P is the resolution of each patch." (Preliminary: Absolute Position Embedding).
- Fixed dimensionality (e.g., strictly 2D) is not explicitly stated beyond image height/width; padding/resizing requirements not specified. Evidence: "H and W are the height and width of the image" (Preliminary: Absolute Position Embedding).

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; example token count shown for DeiT-Ti analysis is 196 tokens. Evidence: "the y-axis represents the number of tokens (196)." (Figure 2).
- Fixed or variable sequence length: Implied by image size and patch size, and experiments use fixed resolutions. Evidence: "N represents the number of patches, calculated as HW/P^2" (Preliminary: Absolute Position Embedding); "All vision transformers are trained on 224×224 resolution images" (Experiment - Image Classification).
- Attention type: Uses multi-head self-attention; global/windowed/hierarchical attention type not explicitly specified for MPVG. Evidence: "Multi-head Self-Attention is denoted as MSA" (Preliminary: Absolute Position Embedding).
- Computational cost management: GAP reduces attention computation with no class token interaction. Evidence: "GAP results in even less computational complexity because it eliminates the need to compute the attention interaction between the class token and the image patches." (Related Work - Class Token & Global Average Pooling).

---

## 8. Positional Encoding (Critical Section)

- Mechanism: Absolute position embedding added to token embeddings. Evidence: "The method of absolute position embedding used in vision transformers is as follows. As shown in Fig 3-(a), PE is added to the token embedding before they are input into the layer." (Preliminary: Absolute Position Embedding).
- Layer-wise delivery: "In Eq. (6), the Layer-wise structure uses independent LN for token embedding(x) and PE. PE is delivered in each layer as follows:" (Preliminary: Layer-wise Structure).
- Applied to input and later layers, and in MPVG to the Last LN: "We combine two structural approaches: (1) adding token embedding and PE before inputting the layer. (2) delivering PE to each layer except the 0th layer." (Maximizing the Position Embedding with GAP). "In MPVG, we modify Eq. (4) as follows ... y = LN(x_{L+1}) + LN'(pos_0)" (Maximizing the Position Embedding with GAP).
- PE compared across alternatives: "Top-1 accuracy comparison with various methods, using DeiT-T, DeiT-S, DeiT-B, Swin-Ti, CeiT-Ti, T2T-ViT-7 on ImageNet-1K." (Table 1) with methods listed as Default, LaPE, PVG, MPVG.

---

## 9. Positional Encoding as a Variable

- Core research variable: "We propose a simple yet effective method called MPVG, which maximizes the effect of PE in the GAP method." (Introduction).
- Multiple positional encodings compared: "Top-1 accuracy comparison with various methods" with Default, LaPE, PVG, MPVG (Table 1); "We compare PVG and MPVG" (Maximizing the Position Embedding with GAP).
- Claim that PE choice is not critical: Not stated.

---

## 10. Evidence of Constraint Masking

- Model sizes reported: "#Params (M)" with values such as "DeiT-Ti ... 5.717" and "DeiT-B ... 86.567" (Table 1); "ViT-Lite ... 3.740" (Table 2).
- Dataset sizes: Not specified (dataset names only).
- Performance gains attributed to PE/MPVG rather than scaling: "we propose MPVG, which maximizes the effectiveness of PE in the GAP approach" (Abstract); "the experimental results show that MPVG outperforms existing methods across vision transformers on various tasks." (Abstract).
- No explicit attribution to scaling model size or dataset size beyond reporting model parameter counts.

---

## 11. Architectural Workarounds

- Global Average Pooling for representation and efficiency: "global average pooling (GAP) has been preferred over the class token method due to its translation-invariant characteristics and superior performance" and "GAP results in even less computational complexity because it eliminates the need to compute the attention interaction between the class token and the image patches." (Related Work - Class Token & Global Average Pooling).
- Removing class token when using GAP: "We remove the class token as we adapt the Global Average Pooling (GAP) method." (Maximizing the Position Embedding with GAP).
- Layer-wise structure with independent LNs and per-layer PE delivery: "each layer has independent Layer Normalizations (LNs) for the token embedding and PE, with PE being gradually delivered across all the layers" (Introduction).
- PVG/MPVG structural changes for PE delivery: "We combine two structural approaches: (1) adding token embedding and PE before inputting the layer. (2) delivering PE to each layer except the 0th layer." (Maximizing the Position Embedding with GAP). "In MPVG, we modify Eq. (4) as follows ... y = LN(x_{L+1}) + LN'(pos_0)" (Maximizing the Position Embedding with GAP).
- Hierarchical PE delivery (excluding layer 0): "we adopt a structure where the token embedding and PE are added before entering layer 0 and a hierarchical structure for delivering PE, excluding layer 0." (Figure 3 caption).

---

## 12. Explicit Limitations and Non-Claims

- Limitation: "MPVG has a potential limitation in that it is incompatible with the class token method." (Conclusion).
- Future work: "we will further explore the broader applicability of MPVG and the effects of PE's counterbalancing as part of our future work." (Conclusion).
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not stated.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Single modality (vision/images) across multiple standard computer vision datasets.
> - Task structure: Supervised evaluation on image classification, object detection, and semantic segmentation benchmarks.
> - Representation rigidity: Patch-based tokenization with fixed training resolutions/crop sizes; example token count 196 for DeiT-Ti.
> - Model sharing vs specialization: ImageNet-1K pretrained backbones reused for detection/segmentation; no joint multi-task training described.
> - Role of positional encoding: Central variable; MPVG modifies PE delivery (per-layer and Last LN) and is extensively ablated.

---

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple tasks within computer vision: "image classification, object detection, and semantic segmentation" (Related Work - Vision Transformers), with experiments on ImageNet-1K/CIFAR-100, COCO 2017, and ADE20K. All evaluations are within the image modality ("field of computer vision") and there is no evidence of multiple modalities or domain generalization claims.
