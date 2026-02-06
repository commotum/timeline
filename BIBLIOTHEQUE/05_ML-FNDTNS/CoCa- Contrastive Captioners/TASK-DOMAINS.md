# CoCa: Contrastive Captioners are Image-Text Foundation Models (Not specified in the paper)
Source: CoCa- Contrastive Captioners.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | Images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Class label (inferred) | 0D (inferred) | Fixed (inferred) |
| Video action recognition | Video frames | 3D (x, y, t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Action class label (inferred) | 0D (inferred) | Fixed (inferred) |
| Image-text retrieval | Images and text | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Retrieved matching image/text items (ranking) (inferred) | 1D (t) (inferred) | Fixed (inferred) |
| Video-text retrieval | Video frames and text captions | 3D (x, y, t) (inferred); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Retrieved matching video/text items (ranking) (inferred) | 1D (t) (inferred) | Fixed (inferred) |
| Image captioning | Images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Text tokens | 1D (t) (inferred) | Not specified in the paper. |
| Visual question answering (VQA) | Images and question text | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Answers | 0D (inferred) | Fixed (inferred) |
| Visual entailment (SNLI-VE) | Images and text | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Entailment label (inferred) | 0D (inferred) | Fixed (inferred) |
| Visual reasoning (NLVR2) | Images and text | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Reasoning label (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
CoCa is described as supporting visual recognition (image classification and video action recognition), crossmodal retrieval (image-text and video-text), multimodal understanding (VQA, SNLI-VE, NLVR2), and image captioning. Inputs span images, video frame sequences, and text, implying 2D and 3D (x, y, t) visual dimensions alongside 1D text (inferred), while outputs include class/decision answers (0D, inferred), retrieval rankings (inferred), and generated text tokens. Dynamics are mostly not specified, with video-text retrieval using 16-frame clips (capped, inferred), and attention/state dynamics are inferred from the encoder-decoder architecture.

## Evidence
### Task: Image classification
- "Our visual recognition experiments are conducted on ImageNet [9] as image recognition benchmark" (Section 4.2.1 Visual Recognition Tasks)
- "We apply a pretrained frozen CoCa model on both image classification and video action recognition." (Section 4.2.1 Visual Recognition Tasks)
- "an attentional pooling is learned together with a softmax cross-entropy loss layer on top of the embedding outputs from CoCa encoder." (Section 4.2.1 Visual Recognition Tasks)
- Inference: In Dimension 2D (x, y), Fixed input dynamics, Static attention, and Constructed state inferred from "CoCa encodes images to latent representations by a neural network encoder" and the fixed pretraining resolution "pretrain with image resolution of  288 × 288  and patch size  18 × 18 , resulting in a total of 256 image tokens"; class-label output/0D/Fixed dynamics inferred from "softmax cross-entropy loss layer". (Sections 3.2, 4.2.1)

### Task: Video action recognition
- "multiple video datasets including Kinetics-400 [57], Kinetics-600 [58], Kinetics-700 [59], Moments-in-Time [60] as test-beds for video action recognition" (Section 4.2.1 Visual Recognition Tasks)
- "We first take multiple frames of a video and feed each frame into the *shared* image encoder individually" (Section 3.3 Contrastive Captioners for Downstream Tasks)
- "For frozenfeature evaluation or finetuning, we learn an additional pooler on top of the spatial and temporal feature tokens with a softmax cross-entropy loss." (Section 3.3 Contrastive Captioners for Downstream Tasks)
- Inference: In Dimension 3D (x, y, t), Static attention, Constructed state, and action-label/0D/Fixed outputs inferred from the use of multiple video frames and the encoder-decoder description ("CoCa encodes images to latent representations by a neural network encoder") plus the "softmax cross-entropy loss" classifier. (Sections 3.2, 3.3)

### Task: Image-text retrieval
- "We evaluate CoCa on the two standard image-text retrieval benchmarks: MSCOCO [63] and Flickr30K [62]." (Section 4.2.2 Crossmodal Alignment Tasks)
- "we first independently feed each image/text to the corresponding encoder and obtain embeddings for all image/text in the test set." (Section 4.2.2 Crossmodal Alignment Tasks)
- "We then retrieve based on cosine similarity scores over the whole test set." (Section 4.2.2 Crossmodal Alignment Tasks)
- Inference: In Dimension 2D (x, y); 1D (t), Static attention, Constructed state, and retrieval ranking outputs (1D) with Fixed dynamics inferred from paired image/text encoding and retrieval over the fixed test set; architectural support from "CoCa encodes images to latent representations by a neural network encoder" and "encode the input text as latent vectors with causally-masked self-attention". (Sections 3.2, 4.2.2)

### Task: Video-text retrieval
- "We evaluate video-text retrieval using CoCa on MSR-VTT [71] using the full split." (Section 4.2.2 Crossmodal Alignment Tasks)
- "For zero-shot video-text retrieval, we use an even simpler approach by computing the mean embedding of 16 frames of the video" (Section 3.3 Contrastive Captioners for Downstream Tasks)
- "We also encode the captions of each video as target embeddings when computing retrieval metrics." (Section 3.3 Contrastive Captioners for Downstream Tasks)
- Inference: In Dimension 3D (x, y, t); 1D (t), Capped input dynamics from "16 frames of the video", Static attention, Constructed state, and retrieval ranking outputs (1D) with Fixed dynamics inferred from retrieval over a test set and the encoder-decoder architecture ("CoCa encodes images to latent representations by a neural network encoder" and "encode the input text as latent vectors with causally-masked self-attention"). (Sections 3.2, 3.3, 4.2.2)

### Task: Image captioning
- "CoCa is also directly applicable to image captioning tasks as an encoder-decoder model." (Section 4.2.3 Image Captioning and Multimodal Understanding Tasks)
- "captioning loss on the multimodal decoder outputs which predicts text tokens autoregressively." (Abstract)
- Inference: In Dimension 2D (x, y), Fixed input dynamics, Static attention, Constructed state, and 1D (t) output dimension inferred from "CoCa encodes images to latent representations by a neural network encoder" and the fixed pretraining resolution "pretrain with image resolution of  288 × 288  and patch size  18 × 18 , resulting in a total of 256 image tokens." (Sections 3.2, Abstract)

### Task: Visual question answering (VQA)
- "the output of encoder-decoder models can jointly encode image and text inputs, and can be used for tasks that require reasoning over both modalities." (Section 4.2.3 Image Captioning and Multimodal Understanding Tasks)
- "visual question answering (VQA v2 [75])" (Section 4.2.3 Image Captioning and Multimodal Understanding Tasks)
- "train linear classifiers on top of the decoder outputs to predict answers" (Section 4.2.3 Image Captioning and Multimodal Understanding Tasks)
- Inference: In Dimension 2D (x, y); 1D (t), Static attention, Constructed state, and 0D/Fixed answer outputs inferred from multimodal image+text encoding and the use of "linear classifiers" for prediction. (Sections 3.2, 4.2.3)

### Task: Visual entailment (SNLI-VE)
- "the output of encoder-decoder models can jointly encode image and text inputs, and can be used for tasks that require reasoning over both modalities." (Section 4.2.3 Image Captioning and Multimodal Understanding Tasks)
- "visual entailment (SNLI-VE [76])" (Section 4.2.3 Image Captioning and Multimodal Understanding Tasks)
- "train linear classifiers on top of the decoder outputs to predict answers" (Section 4.2.3 Image Captioning and Multimodal Understanding Tasks)
- Inference: In Dimension 2D (x, y); 1D (t), Static attention, Constructed state, and 0D/Fixed entailment-label outputs inferred from multimodal image+text encoding and the use of "linear classifiers" for prediction. (Sections 3.2, 4.2.3)

### Task: Visual reasoning (NLVR2)
- "the output of encoder-decoder models can jointly encode image and text inputs, and can be used for tasks that require reasoning over both modalities." (Section 4.2.3 Image Captioning and Multimodal Understanding Tasks)
- "visual reasoning (NLVR2 [77])" (Section 4.2.3 Image Captioning and Multimodal Understanding Tasks)
- "train linear classifiers on top of the decoder outputs to predict answers" (Section 4.2.3 Image Captioning and Multimodal Understanding Tasks)
- Inference: In Dimension 2D (x, y); 1D (t), Static attention, Constructed state, and 0D/Fixed reasoning-label outputs inferred from multimodal image+text encoding and the use of "linear classifiers" for prediction. (Sections 3.2, 4.2.3)
