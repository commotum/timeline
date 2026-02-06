# BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (Not specified in the paper.)
Source: BLIP- Bootstrapping Language-Image Pre-training.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image-text retrieval | images; texts | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | retrieved texts or images (inferred) | 1D (t) (inferred); 2D (x, y) (inferred) | Capped (inferred) |
| image captioning | images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | captions (text) | 1D (t) (inferred) | Capped (inferred) |
| visual question answering (VQA) | image; question | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | answer | 1D (t) (inferred) | Capped (inferred) |
| visual reasoning (NLVR^2) | sentence; pair of images | 1D (t) (inferred); 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | binary label (sentence describes images or not) (inferred) | 0D (inferred) | Fixed (inferred) |
| visual dialog (VisDial) | image; question; dialog history; caption | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | ranked answer candidates (inferred) | 1D (t) (inferred) | Capped (inferred) |
| text-to-video retrieval | text query; video | 1D (t) (inferred); 3D (x, y, t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | retrieved videos (inferred) | 3D (x, y, t) (inferred) | Not specified in the paper. |
| video question answering | video; question | 3D (x, y, t) (inferred); 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | answer (inferred) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
Across downstream tasks, the paper covers image-text retrieval, image captioning, VQA, NLVR^2 visual reasoning, and VisDial, plus zero-shot text-to-video retrieval and video question answering. The modalities span images and text (2D and 1D, inferred) and videos for the video tasks (3D, inferred), with outputs as generated text, binary labels, or retrieved items. Where stated, outputs are capped (caption max length 20; VQA candidate set) and video inputs use fixed sampled frame counts; attention and state dynamics are not specified.

## Evidence
### Task: image-text retrieval
- "We evaluate BLIP for both image-to-text retrieval (TR) and text-to-image retrieval (IR)" (Section 5.1)
- "first select k candidates based on the image-text feature similarity, and then rerank the selected candidates" (Section 5.1)
- Inference: Treated retrieval as operating over images and texts (2D/1D) and producing retrieved items; output dynamics marked capped due to k-candidate reranking. (Section 5.1)

### Task: image captioning
- "generate captions given images" (Section 3.2)
- "set the maximum generation length as 20." (Section A. Downstream Task Details, Image Captioning)
- Inference: Assigned 2D image inputs and 1D text outputs; output dynamics marked capped because a maximum generation length is set. (Section 3.2; Section A. Downstream Task Details)

### Task: visual question answering (VQA)
- "VQA (Antol et al., 2015) requires the model to predict an answer given an image and a question." (Section 5.3)
- "use the decoder to rank the 3,128 candidate answers" (Section A. Downstream Task Details, VQA)
- Inference: Assigned image/question inputs as 2D/1D and answer output as 1D text; output dynamics marked capped because answers are ranked from a fixed candidate set. (Section 5.3; Section A. Downstream Task Details)

### Task: visual reasoning (NLVR^2)
- "NLVR<sup>2</sup> (Suhr et al., 2019) asks the model to predict whether a sentence describes a pair of images." (Section 5.4)
- Inference: Assigned sentence (1D) plus image-pair (2D) inputs and a binary label output (0D) with fixed output dynamics. (Section 5.4)

### Task: visual dialog (VisDial)
- "model needs to predict an answer not only based on the image-question pair, but also considering the dialog history and the image's caption." (Section 5.5)
- "the model ranks a pool of answer candidates" (Section 5.5)
- Inference: Assigned 2D image and 1D text inputs; output treated as a ranked candidate list (1D) with capped dynamics due to a finite answer pool. (Section 5.5)

### Task: text-to-video retrieval
- "zero-shot transfer to *text-to-video retrieval* and *video question answering*" (Section 5.6)
- "uniformly sample n frames per video (n=8) for retrieval" (Section 5.6)
- Inference: Treated inputs as text (1D) and video (3D); input dynamics marked fixed because a fixed number of frames is sampled; output treated as retrieved videos. (Section 5.6)

### Task: video question answering
- "zero-shot transfer to *text-to-video retrieval* and *video question answering*" (Section 5.6)
- "n=16 for QA" (Section 5.6)
- Inference: Treated inputs as video (3D) plus questions (1D) and outputs as answers (1D); input dynamics marked fixed because a fixed number of frames is sampled. (Section 5.6)
