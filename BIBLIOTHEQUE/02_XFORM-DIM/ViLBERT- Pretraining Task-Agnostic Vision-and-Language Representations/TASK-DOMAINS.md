# ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks (Year not specified in the paper.)
Source: ViLBERT- Pretraining Task-Agnostic Vision-and-Language Representations.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Masked multi-modal modelling | Masked words and masked image region inputs | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Reconstructed words and predicted semantic classes for masked image regions | 1D (t); 2D (x, y) (inferred) | Capped (inferred) |
| Multi-modal alignment prediction | Image-text pair (image regions + caption tokens) | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Binary aligned/not-aligned prediction | 0D (inferred) | Fixed (inferred) |
| Visual Question Answering (VQA) | Natural-language question and image | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Answer prediction over 3,129 possible answers | 0D (inferred) | Fixed (inferred) |
| Visual Commonsense Reasoning (VCR) | Image plus question with candidate responses/rationales | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Multiple-choice answer/rationale selection | 0D (inferred) | Fixed (inferred) |
| Grounding Referring Expressions | Natural-language referring expression and image region proposals | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Localized image region (highest-scoring bounding box proposal) | 2D (x, y) (inferred) | Capped (inferred) |
| Caption-Based Image Retrieval | Caption and candidate image pool | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Retrieved image ranking/selection for a caption | 2D (x, y) (inferred) | Capped (inferred) |
| 'Zero-shot' Caption-Based Image Retrieval (diagnostic) | Caption and candidate image pool | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Retrieved image ranking/selection using pretrained alignment score without fine-tuning | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper covers multimodal image-text tasks spanning pretraining proxy objectives and downstream transfer tasks. Inputs consistently combine text sequences and static image content, so the justified input dimensions are 1D (t) and 2D (x, y), while outputs range from 0D decisions (classification/alignment) to 2D localized or retrieved images. Interface behavior is mostly Capped from bounded candidate sets and region proposals, with Fixed outputs for binary or closed-set classification tasks. Attention Dynamic and State Dynamic are inferred from the described transformer/co-attention architecture as Static attention over provided inputs and Constructed internal representations.

## Evidence
### Task: Masked multi-modal modelling
- "In masked multi-modal learning, the model must reconstruct image region categories or words for masked inputs given the observed inputs." (Figure 3 caption, Section 2.2 ViLBERT: Extending BERT to Jointly Represent Images and Text)
- "The masked multi-modal modelling task (shown in Fig. 3a) follows from the masked language modelling task in standard BERT – masking approximately 15% of both words and image region inputs and tasking the model with reconstructing them given the remaining inputs." (Section 2.2 ViLBERT: Extending BERT to Jointly Represent Images and Text)
- Inference: In Dimension and Out Dimension are 1D (t); 2D (x, y) because the task jointly predicts over "words" and "image region" inputs; In/Out Dynamics are Capped from bounded region inputs ("keep between 10 to 36 high-scoring boxes" in Section 3.1); Attention Dynamic is Static and State Dynamic is Constructed based on transformer/co-attention computation over provided inputs (Section 2.2).

### Task: Multi-modal alignment prediction
- "In multi-modal alignment prediction, the model must predict whether or not the caption describes the image content." (Figure 3 caption, Section 2.2 ViLBERT: Extending BERT to Jointly Represent Images and Text)
- "In the multi-modal alignment task (shown in Fig. 3b), the model is presented an image-text pair as  $\{\mathrm{IMG}, v_1, \ldots, v_T, \mathrm{CLS}, w_1, \ldots, w_T, \mathrm{SEP}\}$  and must predict whether the image and text are aligned." (Section 2.2 ViLBERT: Extending BERT to Jointly Represent Images and Text)
- Inference: In Dimension is 1D (t); 2D (x, y) from caption tokens plus image regions; In Dynamics is Capped from bounded region inputs (Section 3.1); Out Dimension is 0D and Out Dynamics is Fixed from the binary aligned/not-aligned decision; Attention Dynamic is Static and State Dynamic is Constructed from the described transformer/co-attention processing (Section 2.2).

### Task: Visual Question Answering (VQA)
- "The VQA task requires answering natural language questions about images." (Section 3.2 Vision-and-Language Transfer Tasks)
- "To fine-tune ViLBERT on VQA, we learn a two layer MLP ... mapping this representation to 3,129 possible answers." (Section 3.2 Vision-and-Language Transfer Tasks)
- Inference: In Dimension is 1D (t); 2D (x, y) from question text plus image; In Dynamics is Capped from bounded region features (Section 3.1); Out Dimension is 0D and Out Dynamics is Fixed because the answer space is an explicit closed set (3,129 answers); Attention Dynamic is Static and State Dynamic is Constructed from the same transformer/co-attention architecture.

### Task: Visual Commonsense Reasoning (VCR)
- "Given an image, the VCR task presents two problems – visual question answering  $(Q \rightarrow A)$  and answer justification  $(QA \rightarrow R)$  – both being posed as multiple-choice problems." (Section 3.2 Vision-and-Language Transfer Tasks)
- "The final prediction is a softmax over these four scores." (Section 3.2 Vision-and-Language Transfer Tasks)
- Inference: In Dimension is 1D (t); 2D (x, y) from language plus image; In Dynamics is Capped due candidate-based formulation and bounded visual proposals; Out Dimension is 0D and Out Dynamics is Fixed from four-way multiple-choice prediction; Attention Dynamic is Static and State Dynamic is Constructed from the transformer/co-attention setup.

### Task: Grounding Referring Expressions
- "The referring expression task is to localize an image region given a natural language reference." (Section 3.2 Vision-and-Language Transfer Tasks)
- "At inference, we use the highest scoring region as the prediction." (Section 3.2 Vision-and-Language Transfer Tasks)
- Inference: In Dimension is 1D (t); 2D (x, y) from text plus image regions; In/Out Dynamics are Capped because the model reranks a finite proposal set; Out Dimension is 2D (x, y) because the output is a localized image region (bounding box); Attention Dynamic is Static and State Dynamic is Constructed from the model architecture.

### Task: Caption-Based Image Retrieval
- "Caption-based image retrieval is the task of identifying an image from a pool given a caption describing its content." (Section 3.2 Vision-and-Language Transfer Tasks)
- "At inference, we score each caption-image pair in the test set and then sort." (Section 3.2 Vision-and-Language Transfer Tasks)
- Inference: In Dimension is 1D (t); 2D (x, y) from captions and images; In/Out Dynamics are Capped from finite candidate pools (including the stated 4-way training setup); Out Dimension is 2D (x, y) because the retrieved object is an image; Attention Dynamic is Static and State Dynamic is Constructed from the transformer/co-attention architecture.

### Task: 'Zero-shot' Caption-Based Image Retrieval (diagnostic)
- "In this 'zero-shot' task, we directly apply the pretrained the multi-modal alignment prediction mechanism to caption-based image retrieval in Flickr30k [26] without fine-tuning (thus the description as 'zero-shot')." (Section 3.2 Vision-and-Language Transfer Tasks)
- "We use the alignment prediction objective as a scoring function and test on the same split as the caption-based image retrieval task described above." (Section 3.2 Vision-and-Language Transfer Tasks)
- Inference: In Dimension is 1D (t); 2D (x, y) and Out Dimension is 2D (x, y) by the same caption-to-image retrieval framing; In/Out Dynamics are Capped from finite retrieval pools; Attention Dynamic is Static and State Dynamic is Constructed from the unchanged pretrained ViLBERT transformer/co-attention pipeline.
