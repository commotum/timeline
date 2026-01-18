## 1. Basic Metadata

Title: "Transformer in Transformer" (Title block)

Authors: "Kai Han An Xiao Enhua Wu Jianyuan Guo Chunjing Xu Yunhe Wang" (Title block)

Year: Year not specified.

Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper proposes "a novel Transformer-iN-Transformer (TNT) architecture for visual recognition" to "enhance the feature representation ability of visual transformers" by dividing images into "visual sentences" and "visual words" (Section 1 Introduction).

---

## 3. Tasks Evaluated

Task name: Image classification (ImageNet pretraining)

Task type: Classification

Dataset(s) used: ImageNet ILSVRC 2012

Domain: natural images

Quotes: "ImageNet ILSVRC 2012 [30] is an image classification benchmark consisting of 1.2M training images belonging to 1000 classes, and 50K validation images with 50 images per class." (Section 3.1 Datasets and Experimental Settings)

Task name: Image classification (CIFAR-10 transfer)

Task type: Classification

Dataset(s) used: CIFAR-10

Domain: natural images

Quotes: "we evaluate our models on 4 image classification datasets" and "These datasets include superordinate-level object classification (CIFAR-10 [18], CIFAR-100 [18])" (Section 3.5 Transfer Learning)

Task name: Image classification (CIFAR-100 transfer)

Task type: Classification

Dataset(s) used: CIFAR-100

Domain: natural images

Quotes: "These datasets include superordinate-level object classification (CIFAR-10 [18], CIFAR-100 [18])" (Section 3.5 Transfer Learning)

Task name: Image classification (Oxford-IIIT Pets transfer)

Task type: Classification

Dataset(s) used: Oxford-IIIT Pets

Domain: natural images

Quotes: "These datasets include ... fine-grained object classification (Oxford-IIIT Pets [26], Oxford 102 Flowers [25] and iNaturalist 2019 [38])" (Section 3.5 Transfer Learning)

Task name: Image classification (Oxford 102 Flowers transfer)

Task type: Classification

Dataset(s) used: Oxford 102 Flowers

Domain: natural images

Quotes: "These datasets include ... fine-grained object classification (Oxford-IIIT Pets [26], Oxford 102 Flowers [25] and iNaturalist 2019 [38])" (Section 3.5 Transfer Learning)

Task name: Image classification (iNaturalist 2019 transfer)

Task type: Classification

Dataset(s) used: iNaturalist 2019

Domain: natural images

Quotes: "These datasets include ... fine-grained object classification (Oxford-IIIT Pets [26], Oxford 102 Flowers [25] and iNaturalist 2019 [38])" (Section 3.5 Transfer Learning)

Task name: Object detection (DETR + TNT)

Task type: Detection

Dataset(s) used: COCO2017

Domain: natural images

Quotes: "Pure Transformer Object Detection. We construct a pure transformer object detection pipeline by combining our TNT and DETR [3]." (Section 3.5 Transfer Learning) and "COCO2017 [22]" (Table 2)

Task name: Semantic segmentation (Trans2Seg + TNT)

Task type: Segmentation

Dataset(s) used: ADE20K

Domain: natural images

Quotes: "Pure Transformer Semantic Segmentation. We adopt the segmentation framework of Trans2Seg [42] to build the pure transformer semantic segmentation based on TNT backbone." (Section 3.5 Transfer Learning) and "ADE20K [49]" (Table 2)

Task name: Object detection (Faster RCNN + TNT)

Task type: Detection

Dataset(s) used: COCO2017 (minival)

Domain: natural images

Quotes: "Object Detection with Faster RCNN" and "The COCO2017 val results are shown in Table 13." (Appendix A.3 Object Detection with Faster RCNN)

---

## 4. Domain and Modality Scope

Is evaluation performed on a single domain? No; it spans multiple natural image datasets. Evidence: "In addition to ImageNet, we also test on the downstream tasks with transfer learning" and "The details of used visual datasets are listed in Table 2." (Section 3.1 Datasets and Experimental Settings)

Is evaluation performed on multiple domains within the same modality? Yes; multiple image datasets/tasks are used within the visual modality (ImageNet, CIFAR-10/100, Oxford 102 Flowers, Oxford-IIIT Pets, iNaturalist 2019, COCO2017, ADE20K). Evidence: "The details of used visual datasets are listed in Table 2." (Section 3.1 Datasets and Experimental Settings)

Is evaluation performed on multiple modalities? Not claimed.

Does the paper claim domain generalization or cross-domain transfer? Yes, within images. Evidence: "we also test on the downstream tasks with transfer learning to evaluate the generalization ability of TNT." (Section 3.1 Datasets and Experimental Settings)

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image classification (ImageNet) | No (trained for this task) | No | Yes (classification head) | "The class token is also used for the subsequent visual recognition task via a fully-connected head." (Section 2.2 Transformer in Transformer) |
| Image classification (CIFAR-10) | Yes (ImageNet pretraining) | Yes | Yes (classification head) | "we transfer TNT-S, TNT-B models trained on ImageNet to the downstream tasks." and "All models are fine-tuned with an image resolution of 384×384." (Section 3.5 Transfer Learning) |
| Image classification (CIFAR-100) | Yes (ImageNet pretraining) | Yes | Yes (classification head) | "we transfer TNT-S, TNT-B models trained on ImageNet to the downstream tasks." and "All models are fine-tuned with an image resolution of 384×384." (Section 3.5 Transfer Learning) |
| Image classification (Oxford-IIIT Pets) | Yes (ImageNet pretraining) | Yes | Yes (classification head) | "we transfer TNT-S, TNT-B models trained on ImageNet to the downstream tasks." and "All models are fine-tuned with an image resolution of 384×384." (Section 3.5 Transfer Learning) |
| Image classification (Oxford 102 Flowers) | Yes (ImageNet pretraining) | Yes | Yes (classification head) | "we transfer TNT-S, TNT-B models trained on ImageNet to the downstream tasks." and "All models are fine-tuned with an image resolution of 384×384." (Section 3.5 Transfer Learning) |
| Image classification (iNaturalist 2019) | Yes (ImageNet pretraining) | Yes | Yes (classification head) | "we transfer TNT-S, TNT-B models trained on ImageNet to the downstream tasks." and "All models are fine-tuned with an image resolution of 384×384." (Section 3.5 Transfer Learning) |
| Object detection (DETR + TNT on COCO2017) | Yes (ImageNet pretraining) | Yes | Yes (DETR detector head) | "Results of object detection on COCO2017 val set with ImageNet pre-training." (Table 10) and "We construct a pure transformer object detection pipeline by combining our TNT and DETR [3]." (Section 3.5 Transfer Learning) |
| Semantic segmentation (Trans2Seg + TNT on ADE20K) | Yes (ImageNet pretraining) | Yes | Yes (Trans2Seg head) | "Results of semantic segmentation on ADE20K val set with ImageNet pre-training." (Table 11) and "We adopt the segmentation framework of Trans2Seg [42] to build the pure transformer semantic segmentation based on TNT backbone." (Section 3.5 Transfer Learning) |
| Object detection (Faster RCNN + TNT on COCO2017) | Yes (ImageNet pretraining) | Yes | Yes (Faster RCNN head) | "Results of Faster RCNN object detection on COCO minival set with ImageNet pre-training." (Table 13) and "We evaluate TNT-S and DeiT-S on Faster RCNN with FPN [21]." (Appendix A.3 Object Detection with Faster RCNN) |

---

## 6. Input and Representation Constraints

2D image assumption: "Given a 2D image, we uniformly split it into n patches" (Section 2.2 Transformer in Transformer).

Fixed patch size: "The patch size is set as 16 × 16." (Section 2.4 Network Architecture)

Fixed sub-patch grid and size: "Each patch is further divided into m sub-patches" and "x^{i,j} ∈ R^{s × s × 3}" where "(s,s) is the spatial size of sub-patches" (Section 2.2 Transformer in Transformer), and "The number of sub-patches is set as m=4 · 4=16 by default." (Section 2.4 Network Architecture)

Fixed token counts for a standard setting: "in the DeiT-S configuration, we have d=384 and n=196. We set c=24 and m=16" (Section 2.3 Complexity Analysis)

Fixed input resolution for main classification experiments: "The corresponding FLOPs for processing a 224 × 224 image" (Section 2.4 Network Architecture)

Fixed input resolution for transfer classification: "All models are fine-tuned with an image resolution of 384×384." (Section 3.5 Transfer Learning)

Variable resizing in detection: "The training images are randomly resized to have a shorter side in the range of [640,800] and a longer side within 1333 pixels. For testing, the shorter side is set as 800 pixels." (Section 3.5 Transfer Learning)

Fixed crop size for segmentation training: "We apply random resize and crop of 512×512 during training." (Section 3.5 Transfer Learning)

---

## 7. Context Window and Attention Structure

Maximum sequence length (explicitly stated for a common setting): "in the DeiT-S configuration, we have d=384 and n=196. We set c=24 and m=16" where n is the sentence-level sequence length and m is the word-level sequence length (Section 2.3 Complexity Analysis).

Sequence length fixed or variable: Variable across experiments due to different input resolutions; e.g., 224×224 for main classification, 384×384 for transfer learning, and resized detection inputs (Sections 2.4 and 3.5).

Attention type: Global self-attention within words and within sentences (hierarchical two-level attention). Evidence: "For the word embeddings, we utilize a transformer block to explore the relation between visual words" and "This outer transformer block T_out is used for modeling relationships among sentence embeddings." (Section 2.2 Transformer in Transformer)

Mechanisms to manage computational cost: The inner word-level attention is shared and kept lightweight. Evidence: "features and attentions between visual words in each visual sentence are calculated independently using a shared network so that the increased amount of parameters and FLOPs (floating-point operations) is negligible." (Section 1 Introduction) and "the increase of FLOPs is small since c ≪ d" (Section 2.3 Complexity Analysis)

---

## 8. Positional Encoding (Critical Section)

Mechanism: Learnable absolute 1D position encodings for both sentences and words. Evidence: "The standard learnable 1D position encodings are utilized here." (Section 2.2 Transformer in Transformer)

Where applied: Added to the input sentence and word embeddings. Evidence: "Z_0 ← Z_0 + E_sentence" and "Y_0^i ← Y_0^i + E_word" (Section 2.2 Transformer in Transformer)

Shared/structured details: "E_word ∈ R^{m × c} are the word position encodings which are shared across sentences." (Section 2.2 Transformer in Transformer)

Fixed vs modified or ablated: Positional encodings are ablated by removal in experiments. Evidence: "We verify their effect by removing them separately." (Section 3.3 Ablation Studies)

---

## 9. Positional Encoding as a Variable

Is positional encoding a core research variable or fixed assumption? It is experimentally varied via ablation. Evidence: "We verify their effect by removing them separately." (Section 3.3 Ablation Studies)

Are multiple positional encodings compared? Only removal ablations are described; no alternative PE types are named. Evidence: "Effect of position encodings... removing them separately." (Section 3.3 Ablation Studies)

Does the paper claim PE choice is "not critical" or secondary? Not stated.

---

## 10. Evidence of Constraint Masking

Model sizes: "There are three variants of TNT networks with different model sizes, namely, TNT-Ti, TNT-S and TNT-B. They consist of 6.1M, 23.8M and 65.6M parameters respectively." (Section 2.4 Network Architecture)

Dataset size: "ImageNet ILSVRC 2012 [30] is an image classification benchmark consisting of 1.2M training images belonging to 1000 classes" (Section 3.1 Datasets and Experimental Settings)

Performance gains attributed to architectural hierarchy/local structure, not scaling data: "indicating the benefit of the introduced TNT framework to preserve local structure information inside the patch." (Section 3.2 TNT on ImageNet) and "With a small increase of computation and memory cost, our TNT block can efficiently model the local structure information and achieve a much better trade-off between accuracy and complexity" (Section 2.3 Complexity Analysis)

Attribution to scaling model size or data: Not explicitly claimed.

---

## 11. Architectural Workarounds

Hierarchical patch and sub-patch structure: "we first divide the input images into several patches as \"visual sentences\" and then further divide them into sub-patches as \"visual words\"." (Section 1 Introduction)

Shared inner transformer for word-level attention: "features and attentions between visual words in each visual sentence are calculated independently using a shared network so that the increased amount of parameters and FLOPs ... is negligible." (Section 1 Introduction)

Fusion of word features into sentence embeddings: "Z_{l-1}^i = Z_{l-1}^i + FC(Vec(Y_l^i))" (Section 2.2 Transformer in Transformer)

Task-specific head for classification: "The class token is also used for the subsequent visual recognition task via a fully-connected head." (Section 2.2 Transformer in Transformer)

Pooling for detection backbone alignment: "add a 2×2 average pooling to make the output size of TNT backbone the same as that of PVT and ResNet." (Section 3.5 Transfer Learning)

Multi-scale feature extraction for Faster RCNN: "We extract the features from different layers of TNT to construct multi-scale features." (Appendix A.3 Object Detection with Faster RCNN)

---

## 12. Explicit Limitations and Non-Claims

Limitation vs state-of-the-art CNNs: "Note that all the transformer-based models are still inferior to EfficientNet which utilizes special depth-wise convolutions, so it is yet a challenge of how to beat EfficientNet using pure transformer." (Section 3.2 TNT on ImageNet)

Potential negative impact: "The potential negative societal impacts may include energy consumption and carbon dioxide emissions of GPU computation." (Section 3.1 Datasets and Experimental Settings)

Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> – Domain scope: multiple natural-image datasets within a single visual modality (ImageNet, CIFAR, COCO, ADE20K).
> – Task structure: classification, detection, and segmentation; downstream tasks evaluated via transfer learning from ImageNet.
> – Representation rigidity: fixed patch size (16 × 16) and fixed sub-patch count (m=16) with mostly fixed input resolutions; detection/segmentation use resizing and fixed crops.
> – Model sharing vs specialization: ImageNet-pretrained TNT backbones are fine-tuned per task with task-specific heads (DETR, Trans2Seg, Faster RCNN).
> – Role of positional encoding: learnable 1D sentence/word encodings added at input and ablated by removal.

---

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple task types, including "image classification," "object detection," and "semantic segmentation" (Sections 3.1 and 3.5), but all evaluations are on natural image datasets within the same visual modality. Transfer learning is described as "downstream tasks with transfer learning to evaluate the generalization ability of TNT" rather than cross-modality or open-world settings (Section 3.1).
