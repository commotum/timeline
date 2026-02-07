# Image as a Foreign Language: BEIT Pretraining for All Vision and Vision-Language Tasks (Not specified in the paper)
Source: Image as a Foreign Language- BEiT Pretraining for All Vision and Vision-Language Tasks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked data modeling (mask-then-predict) | masked images, texts, and image-text pairs | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | masked tokens (text tokens or visual tokens) | 1D (t) (inferred); 2D (x, y) (inferred) | Not specified in the paper. |
| visual question answering | image and natural language question | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | answer label (from candidates) | 0D (inferred) | Not specified in the paper. |
| visual reasoning | pair of images and textual description | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | truth label | 0D (inferred) | Not specified in the paper. |
| image captioning | image | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | natural language caption | 1D (t) (inferred) | Not specified in the paper. |
| image-to-text retrieval | image | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | retrieved text | 1D (t) (inferred) | Not specified in the paper. |
| text-to-image retrieval | text | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | retrieved image | 2D (x, y) (inferred) | Not specified in the paper. |
| object detection | image | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | object detections (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |
| instance segmentation | image | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | instance segmentation masks (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |
| semantic segmentation | image | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | per-pixel labels | 2D (x, y) (inferred) | Not specified in the paper. |
| image classification | image | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | label for an image | 0D (inferred) | Not specified in the paper. |

## Summary
The paper describes a unified masked data modeling pretraining objective over images and texts, and evaluates the model on vision and vision-language tasks including VQA, visual reasoning, captioning, retrieval, detection, segmentation, and classification. The explicit modalities imply 2D (x, y) image inputs and 1D (t) text inputs, with outputs ranging from 0D labels to 1D captions and 2D pixel-wise predictions; these dimensional labels are inferred from the described modalities. Dynamics are not explicitly specified for inputs or outputs, while Attention and State dynamics are inferred as Static and Direct from the described transformer processing and fixed attention masking.

## Evidence
### Task: masked data modeling (mask-then-predict)
- "We pretrain BEIT-3 via a unified masked data modeling [BWDW22] objective on monomodal (i.e., images, and texts) and multimodal data (i.e., image-text pairs)." (Section 2.2 Pretraining Task: Masked Data Modeling)
- "During pretraining, we randomly mask some percentage of text tokens or image patches and train the model to recover the masked tokens." (Section 2.2 Pretraining Task: Masked Data Modeling)
- Inference: In/Out Dimension and Attention/State dynamics are inferred from the stated image/text modalities and masked-token prediction; no dimension/dynamics labels are explicit. (Section 2.2 Pretraining Task: Masked Data Modeling)

### Task: visual question answering
- "The task requires the model to answer natural language questions about input images." (Section 3.1 Vision-Language Downstream Tasks)
- "we conduct finetuning experiments on the VQA v2.0 dataset [GKS+17] and formulate the task as a classification problem." (Section 3.1 Vision-Language Downstream Tasks)
- "The model is trained to predict answers from the 3129 most frequent answer candidates in the training set." (Section 3.1 Vision-Language Downstream Tasks)
- Inference: In Dimension, Attention Dynamic, State Dynamic, and Out Dimension are inferred from the image/text modalities and classification framing; no dimension/dynamics labels are explicit. (Section 3.1 Vision-Language Downstream Tasks)

### Task: visual reasoning
- "The task needs models to perform joint reasoning about images and natural language descriptions." (Section 3.1 Vision-Language Downstream Tasks)
- "determine whether a textual description is true about a pair of images." (Section 3.1 Vision-Language Downstream Tasks)
- Inference: In Dimension, Attention Dynamic, State Dynamic, and Out Dimension are inferred from the image/text modalities and binary decision framing; no dimension/dynamics labels are explicit. (Section 3.1 Vision-Language Downstream Tasks)

### Task: image captioning
- "The task aims to generate a natural language caption for the given image." (Section 3.1 Vision-Language Downstream Tasks)
- "a special self-attention mask is employed for the image captioning task." (Section 3.1 Vision-Language Downstream Tasks)
- Inference: In/Out Dimension and State Dynamic are inferred from the image input and caption text output; Attention Dynamic is inferred as Static from the fixed self-attention mask. (Section 3.1 Vision-Language Downstream Tasks)

### Task: image-to-text retrieval
- "The task is to measure the similarity between images and texts." (Section 3.1 Vision-Language Downstream Tasks)
- "There are two directions depending on the modality of the retrieved target: image-to-text retrieval, and text-to-image retrieval." (Section 3.1 Vision-Language Downstream Tasks)
- Inference: In/Out Dimension, Attention Dynamic, and State Dynamic are inferred from the image/text modalities and retrieval framing; no dimension/dynamics labels are explicit. (Section 3.1 Vision-Language Downstream Tasks)

### Task: text-to-image retrieval
- "The task is to measure the similarity between images and texts." (Section 3.1 Vision-Language Downstream Tasks)
- "There are two directions depending on the modality of the retrieved target: image-to-text retrieval, and text-to-image retrieval." (Section 3.1 Vision-Language Downstream Tasks)
- Inference: In/Out Dimension, Attention Dynamic, and State Dynamic are inferred from the image/text modalities and retrieval framing; no dimension/dynamics labels are explicit. (Section 3.1 Vision-Language Downstream Tasks)

### Task: object detection
- "including image classification, object detection, instance segmentation, and semantic segmentation." (Section 2.1 Backbone Network: Multiway Transformers)
- "for the object detection and instance segmentation tasks." (Section 3.2 Vision Downstream Tasks)
- Inference: Output type and In/Out Dimension, Attention Dynamic, and State Dynamic are inferred from the task name and image modality; no dimension/dynamics labels are explicit. (Sections 2.1 and 3.2)

### Task: instance segmentation
- "including image classification, object detection, instance segmentation, and semantic segmentation." (Section 2.1 Backbone Network: Multiway Transformers)
- "for the object detection and instance segmentation tasks." (Section 3.2 Vision Downstream Tasks)
- Inference: Output type and In/Out Dimension, Attention Dynamic, and State Dynamic are inferred from the task name and image modality; no dimension/dynamics labels are explicit. (Sections 2.1 and 3.2)

### Task: semantic segmentation
- "Semantic segmentation aims to predict the label for each pixel of the given image." (Section 3.2 Vision Downstream Tasks)
- "including image classification, object detection, instance segmentation, and semantic segmentation." (Section 2.1 Backbone Network: Multiway Transformers)
- Inference: In/Out Dimension, Attention Dynamic, and State Dynamic are inferred from the image/pixel formulation; no dimension/dynamics labels are explicit. (Sections 3.2 and 2.1)

### Task: image classification
- "including image classification, object detection, instance segmentation, and semantic segmentation." (Section 2.1 Backbone Network: Multiway Transformers)
- "BEIT-3 is trained as a dual encoder to find the most relevant label for an image." (Section 3.2 Vision Downstream Tasks)
- Inference: In/Out Dimension, Attention Dynamic, and State Dynamic are inferred from the image modality and label output; no dimension/dynamics labels are explicit. (Sections 2.1 and 3.2)
