# Masked-attention Mask Transformer for Universal Image Segmentation (Not specified in the paper.)
Source: Mask2Former- Masked-attention Mask Transformer for Universal Image Segmentation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Panoptic segmentation | Images | 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Panoptic segmentation masks with category + instance labels | 2D (x, y) (inferred) | Capped (inferred) |
| Instance segmentation | Images | 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Instance masks with category labels | 2D (x, y) (inferred) | Capped (inferred) |
| Semantic segmentation | Images | 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Per-pixel semantic label masks | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
Mask2Former targets three image segmentation tasks: panoptic, instance, and semantic segmentation, operating on images and producing pixel-level masks. The paper frames segmentation as grouping pixels and predicting sets of binary masks with category labels, so both inputs and outputs are 2D (x, y) (inferred). Training uses fixed-size crops and bounded resizing, suggesting capped spatial dynamics for input/output (inferred). Masked attention conditioned on predicted masks and learnable query features that generate mask proposals indicate dynamic attention and constructed state (both inferred).

## Evidence
### Task: Panoptic segmentation
- "We present Maskedattention Mask Transformer (Mask2Former), a new architecture capable of addressing any image segmentation task (panoptic, instance or semantic)." (Abstract)
- "Image segmentation groups pixels with different semantics, e.g., category or instance membership." (Abstract)
- "predicting N binary masks, along with N corresponding category labels." (3.1. Mask classification preliminaries)
- "constraining crossattention to within the foreground region of the predicted mask for each query" (3.2. Transformer decoder with masked attention)
- "learnable query features function like a region proposal network [43] and have the ability to generate mask proposals." (3.2.3 Optimization improvements)
- "fixed size crop to  $1024 \times 1024$ ." (4.2. Training settings)
- "resize an image with shorter side to 800 and longer side up-to 1333." (4.2. Training settings)
- Inference: In/Out Dimension set to 2D (x, y); In/Out Dynamics set to Capped; Attention Dynamic set to Dynamic; State Dynamic set to Constructed, based on the quotes above about images/pixels, bounded resizing/crops, masked attention conditioned on predicted masks, and learnable query features that generate mask proposals.

### Task: Instance segmentation
- "We present Maskedattention Mask Transformer (Mask2Former), a new architecture capable of addressing any image segmentation task (panoptic, instance or semantic)." (Abstract)
- "Image segmentation groups pixels with different semantics, e.g., category or instance membership." (Abstract)
- "predicting N binary masks, along with N corresponding category labels." (3.1. Mask classification preliminaries)
- "constraining crossattention to within the foreground region of the predicted mask for each query" (3.2. Transformer decoder with masked attention)
- "learnable query features function like a region proposal network [43] and have the ability to generate mask proposals." (3.2.3 Optimization improvements)
- "fixed size crop to  $1024 \times 1024$ ." (4.2. Training settings)
- "resize an image with shorter side to 800 and longer side up-to 1333." (4.2. Training settings)
- Inference: In/Out Dimension set to 2D (x, y); In/Out Dynamics set to Capped; Attention Dynamic set to Dynamic; State Dynamic set to Constructed, based on the quotes above about images/pixels, bounded resizing/crops, masked attention conditioned on predicted masks, and learnable query features that generate mask proposals.

### Task: Semantic segmentation
- "We present Maskedattention Mask Transformer (Mask2Former), a new architecture capable of addressing any image segmentation task (panoptic, instance or semantic)." (Abstract)
- "Image segmentation groups pixels with different semantics, e.g., category or instance membership." (Abstract)
- "predicting N binary masks, along with N corresponding category labels." (3.1. Mask classification preliminaries)
- "constraining crossattention to within the foreground region of the predicted mask for each query" (3.2. Transformer decoder with masked attention)
- "learnable query features function like a region proposal network [43] and have the ability to generate mask proposals." (3.2.3 Optimization improvements)
- "fixed size crop to  $1024 \times 1024$ ." (4.2. Training settings)
- "resize an image with shorter side to 800 and longer side up-to 1333." (4.2. Training settings)
- Inference: In/Out Dimension set to 2D (x, y); In/Out Dynamics set to Capped; Attention Dynamic set to Dynamic; State Dynamic set to Constructed, based on the quotes above about images/pixels, bounded resizing/crops, masked attention conditioned on predicted masks, and learnable query features that generate mask proposals.
