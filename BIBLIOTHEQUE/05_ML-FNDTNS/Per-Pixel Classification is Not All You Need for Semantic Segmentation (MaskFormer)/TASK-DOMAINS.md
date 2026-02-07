# Per-Pixel Classification is Not All You Need for Semantic Segmentation (Not specified in the paper.)
Source: Per-Pixel Classification is Not All You Need for Semantic Segmentation (MaskFormer).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Semantic segmentation | images | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | semantic segmentation map (class labels per pixel) | 2D (x, y) | Not specified in the paper. |
| Instance-level segmentation | images | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | instance masks with class labels (distinct instances) | 2D (x, y) | Not specified in the paper. |
| Panoptic segmentation | images | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | panoptic segmentation map (thing instances + stuff regions) | 2D (x, y) | Not specified in the paper. |

## Summary
The paper covers semantic, instance-level, and panoptic segmentation of 2D images, producing 2D mask/label outputs. The model description indicates a transformer decoder that attends to image features and produces per-segment embeddings, supporting Static attention and Constructed state (inferred). The paper does not explicitly specify input or output dynamics (Fixed/Capped/Open), so those are marked as not specified.

## Evidence
### Task: Semantic segmentation
- "The goal of semantic segmentation is to partition an image into regions with different semantic categories." (Section 1 Introduction)
- "For semantic segmentation, segments sharing the same category label are merged." (Section 3.4 Mask-classification inference)
- Inference: Attention Dynamic is Static (inferred) and State Dynamic is Constructed (inferred) because "A transformer decoder attends to image features and produces N per-segment embeddings." (Figure 2 caption)

### Task: Instance-level segmentation
- "mask classification is sufficiently general to solve both semantic- and instance-level segmentation tasks" (Abstract)
- "for instance-level segmentation tasks, the index i of the probability-mask pair helps to distinguish different instances of the same class." (Section 3.4 Mask-classification inference)
- Inference: Attention Dynamic is Static (inferred) and State Dynamic is Constructed (inferred) because "A transformer decoder attends to image features and produces N per-segment embeddings." (Figure 2 caption)

### Task: Panoptic segmentation
- "simplifies the landscape of effective approaches to semantic and panoptic segmentation tasks" (Abstract)
- "Finally, to reduce false positive rates in panoptic segmentation we follow previous inference strategies." (Section 3.4 Mask-classification inference)
- Inference: Attention Dynamic is Static (inferred) and State Dynamic is Constructed (inferred) because "A transformer decoder attends to image features and produces N per-segment embeddings." (Figure 2 caption)
