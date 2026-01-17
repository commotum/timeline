## 1. Basic Metadata

- Title: "Masked-attention Mask Transformer for Universal Image Segmentation" (Title header)
- Authors: "Bowen Cheng<sup>1,2*</sup> Ishan Misra<sup>1</sup> Alexander G. Schwing<sup>2</sup> Alexander Kirillov<sup>1</sup> Rohit Girdhar<sup>1</sup>" (Title header)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

"We present Maskedattention Mask Transformer (Mask2Former), a new architecture capable of addressing any image segmentation task (panoptic, instance or semantic)." (Abstract)

## 3. Tasks Evaluated

- Task name: Panoptic segmentation
  - Task type: Segmentation
  - Dataset(s) used: COCO, ADE20K, Cityscapes, Mapillary Vistas
  - Domain: Image datasets; Cityscapes and Mapillary Vistas are urban street-view (COCO/ADE20K domain not specified beyond "image segmentation datasets")
  - Quotes: "Different semantics for grouping pixels, e.g., category or instance membership, have led to different types of segmentation tasks, such as panoptic, instance or semantic segmentation." (1. Introduction) "We evaluate Mask2Former on three image segmentation tasks (panoptic, instance and semantic segmentation) using four popular datasets (COCO [35], Cityscapes [16], ADE20K [65] and Mapillary Vistas [42])." (1. Introduction) "We study Mask2Former using four widely used image segmentation datasets that support semantic, instance and panoptic segmentation: COCO [35] (80 "things" and 53 "stuff" categories), ADE20K [65] (100 "things" and 50 "stuff" categories), Cityscapes [16] (8 "things" and 11 "stuff" categories) and Mapillary Vistas [42] (37 "things" and 28 "stuff" categories)." (4. Experiments - Datasets) "Cityscapes is an urban egocentric street-view dataset with high-resolution images ( $1024 \times 2048$ pixels)." (B.1. Cityscapes) "Mapillary Vistas is a large-scale urban street-view dataset with 18k, 2k and 5k images for training, validation and testing." (B.3. Mapillary Vistas)

- Task name: Instance segmentation
  - Task type: Segmentation
  - Dataset(s) used: COCO, ADE20K, Cityscapes (Mapillary Vistas not used for instance segmentation)
  - Domain: Image datasets; Cityscapes is urban street-view (COCO/ADE20K domain not specified beyond "image segmentation datasets")
  - Quotes: "Different semantics for grouping pixels, e.g., category or instance membership, have led to different types of segmentation tasks, such as panoptic, instance or semantic segmentation." (1. Introduction) "We evaluate Mask2Former on three image segmentation tasks (panoptic, instance and semantic segmentation) using four popular datasets (COCO [35], Cityscapes [16], ADE20K [65] and Mapillary Vistas [42])." (1. Introduction) "We study Mask2Former using four widely used image segmentation datasets that support semantic, instance and panoptic segmentation: COCO [35] (80 "things" and 53 "stuff" categories), ADE20K [65] (100 "things" and 50 "stuff" categories), Cityscapes [16] (8 "things" and 11 "stuff" categories) and Mapillary Vistas [42] (37 "things" and 28 "stuff" categories)." (4. Experiments - Datasets) "We only report panoptic and semantic segmentation results for this dataset." (B.3. Mapillary Vistas) "Cityscapes is an urban egocentric street-view dataset with high-resolution images ( $1024 \times 2048$ pixels)." (B.1. Cityscapes)

- Task name: Semantic segmentation
  - Task type: Segmentation
  - Dataset(s) used: COCO, ADE20K, Cityscapes, Mapillary Vistas
  - Domain: Image datasets; Cityscapes and Mapillary Vistas are urban street-view (COCO/ADE20K domain not specified beyond "image segmentation datasets")
  - Quotes: "Different semantics for grouping pixels, e.g., category or instance membership, have led to different types of segmentation tasks, such as panoptic, instance or semantic segmentation." (1. Introduction) "We evaluate Mask2Former on three image segmentation tasks (panoptic, instance and semantic segmentation) using four popular datasets (COCO [35], Cityscapes [16], ADE20K [65] and Mapillary Vistas [42])." (1. Introduction) "We study Mask2Former using four widely used image segmentation datasets that support semantic, instance and panoptic segmentation: COCO [35] (80 "things" and 53 "stuff" categories), ADE20K [65] (100 "things" and 50 "stuff" categories), Cityscapes [16] (8 "things" and 11 "stuff" categories) and Mapillary Vistas [42] (37 "things" and 28 "stuff" categories)." (4. Experiments - Datasets) "Cityscapes is an urban egocentric street-view dataset with high-resolution images ( $1024 \times 2048$ pixels)." (B.1. Cityscapes) "Mapillary Vistas is a large-scale urban street-view dataset with 18k, 2k and 5k images for training, validation and testing." (B.3. Mapillary Vistas)

## 4. Domain and Modality Scope

- Evaluation scope: Multiple domains within the same modality (images). Evidence: "We evaluate Mask2Former on three image segmentation tasks (panoptic, instance and semantic segmentation) using four popular datasets (COCO [35], Cityscapes [16], ADE20K [65] and Mapillary Vistas [42])." (1. Introduction) "Cityscapes is an urban egocentric street-view dataset with high-resolution images ( $1024 \times 2048$ pixels)." (B.1. Cityscapes) "Mapillary Vistas is a large-scale urban street-view dataset with 18k, 2k and 5k images for training, validation and testing." (B.3. Mapillary Vistas)
- Multiple modalities: Not specified.
- Domain generalization or cross-domain transfer: Claimed across datasets. "Finally we show Mask2Former generalizes beyond the standard benchmarks, obtaining state-of-the-art results on four datasets." (4. Experiments) "It suggests Mask2Former can serve as a universal image segmentation model and results generalize across datasets." (Table 7 / 4.6. Limitations context)

## 5. Model Sharing Across Tasks

Evidence indicates task-specific training despite a shared architecture: "Note, universal architectures are still trained separately for different tasks and datasets, albeit having the same architecture." (1. Introduction) and "Although a single Mask2Former can address any segmentation task, we still need to train it on different tasks." (Table 7. Limitations of Mask2Former)

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Panoptic segmentation | No; trained separately per task (same architecture). | Not specified. | Not specified. | "universal architectures are still trained separately for different tasks and datasets" (1. Introduction); "we still need to train it on different tasks" (Table 7). |
| Instance segmentation | No; trained separately per task (same architecture). | Not specified. | Not specified. | "universal architectures are still trained separately for different tasks and datasets" (1. Introduction); "we still need to train it on different tasks" (Table 7). |
| Semantic segmentation | No; trained separately per task (same architecture). | Not specified. | Not specified. | "universal architectures are still trained separately for different tasks and datasets" (1. Introduction); "we still need to train it on different tasks" (Table 7). |

## 6. Input and Representation Constraints

- Fixed/variable input resolution (training crops): "we use the large-scale jittering (LSJ) augmentation [18,23] with a random scale sampled from range 0.1 to 2.0 followed by a fixed size crop to  $1024 \times 1024$ ." (4.2. Training settings)
- Resizing at inference: "we resize an image with shorter side to 800 and longer side up-to 1333." (4.2. Training settings)
- Cityscapes sizes: "we use a crop size of  $512 \times 1024$ ... During inference, we operate on the whole image ( $1024 \times 2048$ )." (B.1. Cityscapes)
- Mapillary Vistas sizes: "random cropping with a crop size of  $1024 \times 1024$ ... During inference, we resize the longer side to 2048 pixels." (B.3. Mapillary Vistas)
- Multi-scale feature resolutions: "we use the feature pyramid produced by the pixel decoder with resolution 1/32, 1/16 and 1/8 of the original image." (3.2.2 High-resolution features)
- Query/token count (decoder): "We use our Transformer decoder proposed in Section 3.2 with L=3 (*i.e.*, 9 layers total) and 100 queries by default." (4.1. Implementation details) "we use 200 queries for panoptic and instance segmentation models with Swin-L backbone. All other backbones or semantic segmentation models use 100 queries." (B.1. Cityscapes)
- Fixed patch size: Not specified.
- Padding requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; image token count is tied to spatial resolution. " $\mathbf{K}_l, \mathbf{V}_l \in \mathbb{R}^{H_l W_l \times C}$  are the image features" and " $H_l$  and  $W_l$  are the spatial resolution of image features" (3.2.1 Masked attention). "H and W are the original image resolution." (3.2.2 High-resolution features)
- Fixed vs. variable sequence length: Variable with image resolution and scale. " $H_1 = H/32$ ,  $H_2 = H/16$ ,  $H_3 = H/8$  and  $W_1 = W/32$ ,  $W_2 = W/16$ ,  $W_3 = W/8$ , where H and W are the original image resolution." (3.2.2 High-resolution features)
- Attention type: Masked (sparse/local) cross-attention plus self-attention. "masked attention ... constraining crossattention to within the foreground region of the predicted mask for each query, instead of attending to the full feature map." (3.2 Transformer decoder with masked attention)
- Mechanisms to manage computational cost: "we propose an efficient multi-scale strategy to introduce high-resolution features while controlling the increase in computation." (3.2.2 High-resolution features) "feed one resolution of the multi-scale feature to one Transformer decoder layer at a time." (3.2.2 High-resolution features)

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Sinusoidal positional embeddings for image features plus learnable scale-level embeddings; learnable query positional embeddings. "For each resolution, we add both a sinusoidal positional embedding  $e_{\rm pos}$ ... and a learnable scale-level embedding  $e_{\rm lvl}$" (3.2.2 High-resolution features). "query features ... are associated with learnable positional embeddings." (3.2.3 Optimization improvements). "query positional embeddings are added to query features in every Transformer decoder layer" (C.3. Object query analysis).
- Where applied: Image feature pyramid (per resolution) and queries at every decoder layer. "For each resolution, we add both a sinusoidal positional embedding ... and a learnable scale-level embedding" (3.2.2). "query positional embeddings are added to query features in every Transformer decoder layer when computing the attention weights." (C.3. Object query analysis)
- Fixed across experiments / modified per task / ablated: Not specified.

## 9. Positional Encoding as a Variable

- Core research variable vs fixed assumption: Treated as a fixed architectural choice; no PE-focused experiments described. "For each resolution, we add both a sinusoidal positional embedding ... and a learnable scale-level embedding" (3.2.2). "query features ... are associated with learnable positional embeddings." (3.2.3)
- Multiple positional encodings compared: Not specified.
- PE claimed as not critical or secondary: Not specified.

## 10. Evidence of Constraint Masking

- Model sizes: "Mask2Former (ours) | R101 | 100 queries | 50 | 52.6 | 58.5 | 43.7 | 42.6 | 62.4 | 63M | 293G | 7.2" (Table 1). "Mask2Former (ours) | R50 | 100 queries | 50 | 51.9 | 57.7 | 43.0 | 41.7 | 61.7 | 44M | 226G | 8.6" (Table 1)
- Dataset sizes: "Cityscapes is an urban egocentric street-view dataset with high-resolution images ( $1024 \times 2048$  pixels). It contains 2975 images for training, 500 images for validation and 1525 images for testing with a total of 19 classes." (B.1. Cityscapes) "Mapillary Vistas is a large-scale urban street-view dataset with 18k, 2k and 5k images for training, validation and testing." (B.3. Mapillary Vistas)
- Primary attribution of gains: Architectural and training changes, not explicit scaling of model size or data. "First, we use masked attention in the Transformer decoder ... our masked attention leads to faster convergence and improved performance. Second, we use *multi-scale high-resolution features* ... Third, we propose optimization improvements ... These improvements not only boost the model performance, but also make training significantly easier" (1. Introduction). "We thus completely remove dropout in our decoder." (3.2.3) "This new training strategy effectively reduces training memory by  $3\times$ , from 18GB to 6GB per image" (3.3 Improving training efficiency).

## 11. Architectural Workarounds

- Masked cross-attention to localize computation: "masked attention ... constraining crossattention to within the foreground region of the predicted mask for each query, instead of attending to the full feature map." (3.2 Transformer decoder with masked attention)
- Efficient multi-scale feature usage to control compute: "we propose an efficient multi-scale strategy to introduce high-resolution features while controlling the increase in computation." (3.2.2 High-resolution features)
- Feature pyramid with fixed scales: "feature pyramid ... resolution 1/32, 1/16 and 1/8 of the original image." (3.2.2 High-resolution features)
- Decoder optimization tweaks: "we switch the order of self- and cross-attention ... make query features learnable ... and remove dropout" (3.2.3 Optimization improvements)
- Point-sampled mask loss to reduce memory: "we calculate the mask loss with sampled points ... This new training strategy effectively reduces training memory by  $3\times$ , from 18GB to 6GB per image" (3.3 Improving training efficiency)

## 12. Explicit Limitations and Non-Claims

- Task-specific training required: "Although a single Mask2Former can address any segmentation task, we still need to train it on different tasks." (Table 7. Limitations of Mask2Former)
- Limited cross-task training: "This suggests that even though Mask2Former can generalize to different tasks, it still needs to be trained for those specific tasks." (4.6. Limitations)
- Future work (single model across tasks/datasets): "In the future, we hope to develop a model that can be trained only once for multiple tasks and even for multiple datasets." (4.6. Limitations)
- Small-object and multiscale limitations: "Mask2Former struggles with segmenting small objects and is unable to fully leverage multiscale features." (4.6. Limitations)

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Multiple image datasets in the same modality; includes urban street-view datasets (Cityscapes, Mapillary Vistas).
> - Task structure: Panoptic, instance, and semantic segmentation only.
> - Representation rigidity: Fixed training crops (e.g.,  $1024 \times 1024$ ), multi-scale 1/32-1/8 feature pyramid, fixed query counts (100/200).
> - Model sharing vs specialization: Same architecture but trained separately per task/dataset.
> - Role of positional encoding: Sinusoidal + learnable embeddings applied to features/queries; no PE variations reported.

### 14. Final Classification

**Multi-task, multi-domain (constrained).** The paper evaluates three segmentation tasks "(panoptic, instance and semantic segmentation)" on four datasets (COCO, Cityscapes, ADE20K, Mapillary Vistas), indicating multi-task evaluation across multiple image domains (1. Introduction). It also states that it "still need[s] to train it on different tasks," so task coverage is constrained by separate task-specific training rather than a single shared model (Table 7 / 4.6).
