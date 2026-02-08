# ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision (2021)
Source: ViLT- Vision-and-Language Transformer Without Convolution or Region Supervision.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image-text matching (ITM) | image-text pairs (image patches and text tokens) | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | binary match/non-match label | 0D (inferred) | Fixed |
| Masked language modeling (MLM) | masked text tokens with aligned image patches | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | predicted vocabulary labels for masked tokens | 1D (t) (inferred) | Capped (inferred) |
| Visual question answering (VQAv2 classification) | image and natural-language question | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | answer class (3,129 classes) | 0D (inferred) | Fixed |
| Natural language visual reasoning (NLVR2 binary classification) | two images and natural-language question | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | binary prediction | 0D (inferred) | Fixed |
| Image-to-text retrieval | query image and candidate texts | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | ranked text candidates | 1D (t) (inferred) | Capped |
| Text-to-image retrieval | query text and candidate images | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | ranked image candidates | 1D (t) (inferred) | Capped |

## Summary
The paper covers both pre-training and downstream vision-language tasks: image-text matching, masked language modeling, classification (VQAv2 and NLVR2), and bidirectional retrieval. Across tasks, inputs combine 2D image content and 1D token sequences, while outputs are either 0D class decisions or 1D ordered/token outputs. Based on explicit token limits (e.g., maximum sampled patches and top-K retrieval evaluation), the task interfaces are capped, with fixed output dynamics for classification tasks and capped output dynamics for MLM/retrieval. The model uses a predefined concatenated multimodal token sequence (Static attention, inferred) and acts as a direct reactive mapper without explicit persistent constructed state (Direct, inferred).

## Evidence
### Task: Image-text matching (ITM)
- "We train ViLT with two objectives commonly used to train VLP models: image text matching (ITM) and masked language modeling (MLM)." (Section 3.2 Pre-training Objectives)
- "**Image Text Matching.** We randomly replace the aligned image with a different image with the probability of 0.5. A single linear layer ITM head projects the pooled output feature p to logits over binary class, and we compute negative log-likelihood loss as our ITM loss." (Section 3.2 Pre-training Objectives)
- Inference: In Dimension is 2D (x, y); 1D (t) from "text  $t \in \mathbb{R}^{L \times |V|}$" and "The input image  $I \in \mathbb{R}^{C \times H \times W}$" (Section 3.1 Model Overview). In Dynamics is Capped from "Patch projection of ViLT-B/32 yields ... 240 patches" and "we sample 200 patches at maximum during pre-training" (Section 4.2 Implementation Details). Attention Dynamic is Static because text and image are "concatenated into a combined sequence  $z^0$" processed by transformer layers (Section 3.1 Model Overview). State Dynamic is Direct because ITM prediction is computed directly from pooled representation p with a linear head (Section 3.2 Pre-training Objectives). Out Dimension 0D is inferred from binary class output.

### Task: Masked language modeling (MLM)
- "**Masked Language Modeling.** This objective is to predict the ground truth labels of masked text tokens  $t_{\rm masked}$  from its contextualized vector  $z_{\rm masked}^D|_t$ ." (Section 3.2 Pre-training Objectives)
- "We use a two-layer MLP MLM head that inputs  $z_{\mathrm{masked}}^D|_t$  and outputs logits over vocabulary, just as the MLM objective of BERT." (Section 3.2 Pre-training Objectives)
- Inference: In Dimension uses text plus aligned image streams from the shared multimodal setup in Section 3.1 ("text  $t \in \mathbb{R}^{L \times |V|}$"; "The input image  $I \in \mathbb{R}^{C \times H \times W}$"). In Dynamics is Capped using the same capped patch pipeline (Section 4.2 Implementation Details). Attention Dynamic is Static from the fixed concatenated sequence processing in Section 3.1. State Dynamic is Direct because MLM outputs are produced from contextualized token vectors via an MLP head without persistent external state (Section 3.2). Out Dimension 1D (t) and Out Dynamics Capped are inferred because masked-token outputs follow token positions within bounded input length.

### Task: Visual question answering (VQAv2 classification)
- "**Visual Question Answering.** The VQAv2 task asks for answers given pairs of an image and a question in natural language." (Section 4.3 Classification Tasks)
- "The annotated answers are originally in free-form natural language, but it is a common practice to convert the task to a classification task with 3,129 answer classes." (Section 4.3 Classification Tasks)
- Inference: In Dimension is 2D (x, y); 1D (t) from image-plus-question inputs (Section 4.3) and the model's image/text tokenization (Section 3.1). In Dynamics is Capped from the shared image patch cap (Section 4.2). Attention Dynamic is Static from the fixed multimodal token sequence processing (Section 3.1). State Dynamic is Direct because the downstream head maps pooled features to class output (Sections 4.3 and 3.1). Out Dimension 0D is inferred from single-class prediction.

### Task: Natural language visual reasoning (NLVR2 binary classification)
- "Natural Language for Visual Reasoning. The NLVR2 task is a binary classification task given triplets of two images and a question in natural language." (Section 4.3 Classification Tasks)
- "The head takes the concatenation of two pooled representations (p) as input and outputs the binary prediction." (Section 4.3 Classification Tasks)
- Inference: In Dimension is 2D (x, y); 1D (t) from two images plus one text query (Section 4.3). In Dynamics is Capped from the same capped visual tokenization pipeline applied across experiments (Section 4.2). Attention Dynamic is Static from fixed-sequence transformer processing (Section 3.1). State Dynamic is Direct because prediction is made directly from pooled representations with a task head (Section 4.3). Out Dimension 0D is inferred from binary output.

### Task: Image-to-text retrieval
- "For image-to-text and text-to-image retrieval, we measure both zero-shot and fine-tuned performance<sup>8</sup>." (Section 4.4 Retrieval Tasks)
- "<sup>&</sup>lt;sup>8</sup>R@K corresponds to whether the ground truth is included among top K results from the validation set." (Section 4.4 Retrieval Tasks, footnote 8)
- Inference: In Dimension is 2D (x, y); 1D (t) from image-text pair scoring (Section 4.4 with Section 3.1 multimodal inputs). In Dynamics is Capped from bounded patch counts (Section 4.2) and finite validation ranking for R@K (Section 4.4). Attention Dynamic is Static from fixed concatenated-sequence transformer computation (Section 3.1). State Dynamic is Direct because retrieval scores come from the similarity/ITM-based head on pair representations (Section 4.4). Out Dimension 1D (t) is inferred because retrieval output is an ordered ranking.

### Task: Text-to-image retrieval
- "For image-to-text and text-to-image retrieval, we measure both zero-shot and fine-tuned performance<sup>8</sup>." (Section 4.4 Retrieval Tasks)
- "We report the zero shot retrieval results in Table 3 and finetuned results in Table 4." (Section 4.4 Retrieval Tasks)
- Inference: In Dimension is 1D (t); 2D (x, y) from text-query/image-candidate retrieval in the same multimodal scoring setup (Section 4.4 with Section 3.1). In Dynamics is Capped from bounded visual tokens and finite benchmark candidate sets (Sections 4.2 and 4.4). Attention Dynamic is Static from fixed-sequence transformer processing (Section 3.1). State Dynamic is Direct because outputs are ranked directly from pairwise similarity scores without persistent constructed memory (Section 4.4). Out Dimension 1D (t) is inferred as ordered ranking output.
