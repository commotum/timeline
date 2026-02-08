# SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers (Not specified in the paper)
Source: SegFormer- Simple and Efficient Design for Semantic Segmentation with Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Semantic segmentation | Images | 2D (x, y) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Semantic segmentation mask (per-pixel class labels) | 2D (x, y) | Capped (inferred) |

## Summary
The paper covers a single task: semantic segmentation on natural images, evaluated on ADE20K, Cityscapes, COCO-Stuff, and Cityscapes-C corruptions. The task maps 2D image grids to 2D segmentation masks. The OCR supports variable test resolutions but finite crop/windowed processing, so input/output dynamics are Capped (inferred). The encoder-decoder pipeline uses fixed processing over the given image while constructing multi-level internal features, supporting Static attention and Constructed state (both inferred).

## Evidence
### Task: Semantic segmentation
- "Semantic segmentation is a fundamental task in computer vision and enables many downstream applications." (Section 1 Introduction)
- "Given an image of size  $H\times W\times 3$ , we first divide it into patches of size  $4\times 4$ ." (Section 3 Method)
- "We then pass these multi-level features to the All-MLP decoder to predict the segmentation mask at a  $\frac{H}{4}\times \frac{W}{4}\times N_{cls}$  resolution" (Section 3 Method)
- "our encoder can easily adapt to arbitrary test resolutions without impacting the performance." (Section 1 Introduction)
- Inference: `In Dynamics` and `Out Dynamics` are marked as `Capped (inferred)` because the paper states variable resolution handling ("arbitrary test resolutions") but still processes finite image/crop/window sizes (e.g., "random cropping to  $512 \times 512$ ,  $1024 \times 1024$ ,  $512 \times 512$ ..." and "sliding window test by cropping  $1024 \times 1024$  windows," Section 4.1 Experimental Settings). `Attention Dynamic` is `Static (inferred)` because the model applies a fixed encoder-decoder computation to the provided image rather than runtime retrieval/selection outside that input. `State Dynamic` is `Constructed (inferred)` because the method explicitly builds and fuses internal multi-level features ("obtain multi-level features ..." and decoder fusion in Section 3 Method).
