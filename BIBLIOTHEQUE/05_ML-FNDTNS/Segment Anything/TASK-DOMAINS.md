# Segment Anything (2023)
Source: Segment Anything.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| promptable segmentation | images with segmentation prompts (points, boxes, masks, free-form text) | 2D (x, y); 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | valid segmentation mask(s) | 2D (x, y) (inferred) | Capped (inferred) |
| automatic mask generation for dataset annotation | images with regular-grid foreground point prompts | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | automatically generated object masks per image | 2D (x, y) (inferred) | Capped (inferred) |
| single-point object segmentation | image with a single foreground point prompt | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | object mask (most confident among predicted masks) | 2D (x, y) (inferred) | Capped (inferred) |
| edge detection | image with 16 x 16 regular-grid foreground point prompts | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | edge map | 2D (x, y) (inferred) | Capped (inferred) |
| object proposal generation (segment everything) | image with regular-grid foreground point prompts | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | mask proposals (up to 1000 per image) | 2D (x, y) (inferred) | Capped (inferred) |
| instance segmentation | image with detector output box prompts | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | instance masks for detected objects | 2D (x, y) (inferred) | Capped (inferred) |
| text-to-mask segmentation | image with free-form text prompt (CLIP text embedding) | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | segmentation mask from text prompt | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper frames SAM around a core promptable segmentation task, then demonstrates transfer to automatic annotation, single-point segmentation, edge detection, object proposals, instance segmentation, and text-to-mask. Inputs are image-centered with geometric prompts, and one task adds free-form text prompts; outputs are 2D spatial maps (masks or edges), often as one or a capped set. From the OCR evidence, the paper supports 2D (x, y) processing broadly, plus 1D (t) text input in text-prompted settings. Attention and state labels are inferred from the architecture description: prompt/image cross-attention over provided inputs and reusable image embeddings across prompts.

## Evidence
### Task: promptable segmentation
- "The promptable segmentation task, then, is to return a valid segmentation mask given any prompt." (Section 2, Task)
- "Task. We start by translating the idea of a prompt from NLP to segmentation, where a prompt can be a set of foreground / background points, a rough box or mask, free-form text, or, in general, any information indicating what to segment in an image." (Section 2, Task)
- Inference: `In Dimension = 2D (x, y); 1D (t)` is inferred from image prompts plus "free-form text"; `In Dynamics = Open` is inferred from "any prompt" and iterative prompting ("simulate an interactive setup ... in 11 rounds per mask," Section 3). `Attention Dynamic = Static` is inferred because attention is over provided prompt/image tokens (Section 3, Mask decoder; Section A, Lightweight mask decoder). `State Dynamic = Constructed` is inferred from reusable embeddings ("the same image embedding can be reused," Section 1 Model; "image encoder runs once per image," Section 3).

### Task: automatic mask generation for dataset annotation
- "In the final stage, we prompt SAM with a regular grid of foreground points, yielding on average ~100 high-quality masks per image." (Section 1, Data engine)
- "Specifically, we prompted the model with a 32×32 regular grid of points and for each point predicted a set of masks that may correspond to valid objects." (Section 4, Fully automatic stage)
- Inference: `In Dimension = 2D (x, y)` and `Out Dimension = 2D (x, y)` are inferred from point prompts over images and mask outputs. `In Dynamics = Fixed` is inferred from fixed regular-grid prompting and fixed crop-grid recipes (Section 4; Section B, Cropping). `Out Dynamics = Capped` is inferred from confidence/stability filtering and NMS-based duplicate suppression (Section 4; Section B).

### Task: single-point object segmentation
- "We evaluate segmenting an object from a *single* foreground point." (Section 7.1, Task)
- "Since SAM is capable of predicting multiple masks, we evaluate only the model's most confident mask by default." (Section 7.1)
- Inference: `In Dimension = 2D (x, y)` and `Out Dimension = 2D (x, y)` are inferred from point-on-image input and mask output. `In Dynamics = Fixed` is inferred from the explicit single-point prompt. `Out Dynamics = Capped` is inferred from SAM's bounded multi-mask output design (Section 3, Resolving ambiguity; Section A, Making the model ambiguity-aware).

### Task: edge detection
- "We evaluate SAM on the classic low-level task of edge detection using BSDS500 [72, 3]." (Section 7.2, Approach)
- "Specifically, we prompt SAM with a  $16 \times 16$  regular grid of foreground points resulting in 768 predicted masks ... Then, edge maps are computed using Sobel filtering of unthresholded mask probability maps ..." (Section 7.2, Approach)
- Inference: `In Dimension = 2D (x, y)` and `Out Dimension = 2D (x, y)` are inferred from image-space prompting and edge-map output. `In Dynamics = Fixed` is inferred from the explicit `16 × 16` grid; `Out Dynamics = Capped` is inferred because one edge map is produced per image after a fixed postprocessing pipeline. `Attention Dynamic = Static` and `State Dynamic = Constructed` follow the same SAM architectural evidence as above (Section 3; Section A).

### Task: object proposal generation (segment everything)
- "Next, we evaluate SAM on the mid-level task of object proposal generation [2, 102]." (Section 7.3, Approach)
- "We choose a  $64 \times 64$  point grid and an NMS threshold of 0.9, which produces  $\sim 900$  masks per image on average. At evaluation, if greater than 1000 masks have been proposed in an image, they are ranked ... then truncated to the top 1000 proposals." (Section D.3, Method)
- Inference: `In Dimension = 2D (x, y)` and `Out Dimension = 2D (x, y)` are inferred from point-on-image inputs and mask proposals. `In Dynamics = Fixed` is inferred from fixed grid prompting in the described method. `Out Dynamics = Capped` is inferred from explicit truncation to top-1000 proposals.

### Task: instance segmentation
- "Moving to higher-level vision, we use SAM as the segmentation module of an instance segmenter." (Section 7.4, Approach)
- "The implementation is simple: we run a object detector (the ViTDet used before) and prompt SAM with its output boxes." (Section 7.4, Approach)
- Inference: `In Dimension = 2D (x, y)` and `Out Dimension = 2D (x, y)` are inferred from box prompts and mask outputs on images. `In Dynamics = Capped` and `Out Dynamics = Capped` are inferred because detector outputs a variable but finite set of boxes/masks per image. `Attention Dynamic = Static` and `State Dynamic = Constructed` are inferred from the same SAM architecture evidence (Section 3; Section A).

### Task: text-to-mask segmentation
- "Finally, we consider an even higher-level task: segmenting objects from free-form text." (Section 7.5, Approach)
- "That is, at inference time we run text through CLIP's text encoder and then give the resulting text embedding as a prompt to SAM." (Section 7.5, Approach)
- Inference: `In Dimension = 2D (x, y); 1D (t)` is inferred from image-plus-text prompting. `In Dynamics = Capped` is inferred from encoder-based text prompting and single-prompt inference setup as described in Section 7.5 and Section D.5. `Out Dimension = 2D (x, y)` and `Out Dynamics = Capped` are inferred from mask prediction behavior of SAM. `Attention Dynamic = Static` and `State Dynamic = Constructed` are inferred from the same architecture text (Section 3; Section A).
