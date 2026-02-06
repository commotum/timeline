# Align before Fuse: Vision and Language Representation Learning with Momentum Distillation (Not specified in the paper.)
Source: ALBEF- Align Before Fuse.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image-text contrastive learning (ITC) | image; text | 2D (x, y); 1D (t) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | image-text similarity score / probability | 0D | Not specified in the paper. |
| Masked language modeling (MLM) | image; masked text | 2D (x, y); 1D (t) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | masked word tokens | 1D (t) | Not specified in the paper. |
| Image-text matching (ITM) | image; text | 2D (x, y); 1D (t) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | match label (matched vs not matched) | 0D | Not specified in the paper. |
| Image-to-text retrieval | image | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | text descriptions (ranked) | 1D (t) | Not specified in the paper. |
| Text-to-image retrieval | text | 1D (t) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | images (ranked) | 2D (x, y) | Not specified in the paper. |
| Visual entailment (SNLI-VE) | image; text | 2D (x, y); 1D (t) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | entailment label (entailment/neutral/contradictory) | 0D | Not specified in the paper. |
| Visual question answering (VQA) | image; question text | 2D (x, y); 1D (t) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | answer text | 1D (t) | Not specified in the paper. |
| Natural language for visual reasoning (NLVR^2) | image pair; text | 2D (x, y); 1D (t) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | classification label (text describes pair or not) | 0D | Not specified in the paper. |
| Text-assignment (TA) | image pair; text | 2D (x, y); 1D (t) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | assignment label (first/second/none) | 0D | Not specified in the paper. |
| Visual grounding (weakly-supervised) | image; textual description | 2D (x, y); 1D (t) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | localized region / heatmap / ranked proposals | 2D (x, y) | Not specified in the paper. |

## Summary
The paper covers multimodal tasks spanning pre-training (image-text contrastive learning, masked language modeling, image-text matching) and downstream vision-language tasks including retrieval, visual entailment, VQA, NLVR^2, text-assignment pre-training, and visual grounding. Inputs are images and text (including image pairs for NLVR^2 and TA), with outputs ranging from 0D labels/scores to 1D text tokens and 2D image regions or images. Dimensions are primarily 2D (images) and 1D (text), while input/output dynamics are not specified. Attention and state dynamics are inferred from the described encoder and cross-attention architecture.

## Evidence
### Task: Image-text contrastive learning (ITC)
- "We pre-train ALBEF with three objectives: image-text contrastive learning (ITC)" (Section 3.2 Pre-training Objectives)
- "Image-Text Contrastive Learning aims to learn better unimodal representations before fusion." (Section 3.2 Pre-training Objectives)
- Inference: Attention Dynamic = Static and State Dynamic = Constructed because the model uses "an image encoder, a text encoder, and a multimodal encoder" and "cross attention at each layer" (Section 3.1 Model Architecture).

### Task: Masked language modeling (MLM)
- "**Masked Language Modeling** utilizes both the image and the contextual text to predict the masked words." (Section 3.2 Pre-training Objectives)
- "We randomly mask out the input tokens with a probability of 15%" (Section 3.2 Pre-training Objectives)
- Inference: Attention Dynamic = Static and State Dynamic = Constructed because the model uses "an image encoder, a text encoder, and a multimodal encoder" and "cross attention at each layer" (Section 3.1 Model Architecture).

### Task: Image-text matching (ITM)
- "**Image-Text Matching** predicts whether a pair of image and text is positive (matched) or negative (not matched)." (Section 3.2 Pre-training Objectives)
- "append a fully-connected (FC) layer followed by softmax to predict a two-class probability" (Section 3.2 Pre-training Objectives)
- Inference: Attention Dynamic = Static and State Dynamic = Constructed because the model uses "an image encoder, a text encoder, and a multimodal encoder" and "cross attention at each layer" (Section 3.1 Model Architecture).

### Task: Image-to-text retrieval
- "Image-Text Retrieval contains two subtasks: image-to-text retrieval (TR) and text-to-image retrieval (IR)." (Section 5 Downstream V+L Tasks)
- "During inference, we first compute the feature similarity score $s_{\rm itc}$ for all image-text pairs." (Section 5 Downstream V+L Tasks)
- Inference: Attention Dynamic = Static and State Dynamic = Constructed because the model uses "an image encoder, a text encoder, and a multimodal encoder" and "cross attention at each layer" (Section 3.1 Model Architecture).

### Task: Text-to-image retrieval
- "Image-Text Retrieval contains two subtasks: image-to-text retrieval (TR) and text-to-image retrieval (IR)." (Section 5 Downstream V+L Tasks)
- "During inference, we first compute the feature similarity score $s_{\rm itc}$ for all image-text pairs." (Section 5 Downstream V+L Tasks)
- Inference: Attention Dynamic = Static and State Dynamic = Constructed because the model uses "an image encoder, a text encoder, and a multimodal encoder" and "cross attention at each layer" (Section 3.1 Model Architecture).

### Task: Visual entailment (SNLI-VE)
- "**Visual Entailment** (SNLI-VE<sup>5</sup> [51]) is a fine-grained visual reasoning task" (Section 5 Downstream V+L Tasks)
- "predict whether the relationship between an image and a text is entailment, neutral, or contradictory." (Section 5 Downstream V+L Tasks)
- Inference: Attention Dynamic = Static and State Dynamic = Constructed because the model uses "an image encoder, a text encoder, and a multimodal encoder" and "cross attention at each layer" (Section 3.1 Model Architecture).

### Task: Visual question answering (VQA)
- "**Visual Question Answering** (VQA [52]) requires the model to predict an answer given an image and a question." (Section 5 Downstream V+L Tasks)
- "we use a 6-layer transformer decoder to generate the answer." (Section 5 Downstream V+L Tasks)
- Inference: Attention Dynamic = Static and State Dynamic = Constructed because the model uses "an image encoder, a text encoder, and a multimodal encoder" and "cross attention at each layer" (Section 3.1 Model Architecture).

### Task: Natural language for visual reasoning (NLVR^2)
- "Natural Language for Visual Reasoning (NLVR<sup>2</sup> [19]) requires the model to predict whether a text describes a pair of images." (Section 5 Downstream V+L Tasks)
- "the two blocks receive two sets of image embeddings for the image pair." (Section 5 Downstream V+L Tasks)
- Inference: Attention Dynamic = Static and State Dynamic = Constructed because the model uses "an image encoder, a text encoder, and a multimodal encoder" and "cross attention at each layer" (Section 3.1 Model Architecture).

### Task: Text-assignment (TA)
- "We design a text-assignment (TA) task as follows: given a pair of images and a text" (Section 5 Downstream V+L Tasks)
- "assign the text to either the first image, the second image, or none of them." (Section 5 Downstream V+L Tasks)
- Inference: Attention Dynamic = Static and State Dynamic = Constructed because the model uses "an image encoder, a text encoder, and a multimodal encoder" and "cross attention at each layer" (Section 3.1 Model Architecture).

### Task: Visual grounding (weakly-supervised)
- "**Visual Grounding** aims to localize the region in an image that corresponds to a specific textual description." (Section 5 Downstream V+L Tasks)
- "we extend Grad-CAM [9] to acquire heatmaps, and use them to rank the detected proposals" (Section 5 Downstream V+L Tasks)
- Inference: Attention Dynamic = Static and State Dynamic = Constructed because the model uses "an image encoder, a text encoder, and a multimodal encoder" and "cross attention at each layer" (Section 3.1 Model Architecture).
