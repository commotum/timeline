# UNITER: UNiversal Image-TExt Representation Learning (Not specified in the paper.)
Source: UNITER- Universal Image-Text Representation Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Masked language modeling (MLM) | Text tokens with masked positions + image regions | 1D (t) (inferred); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Masked word/token predictions | 1D (t) (inferred) | Capped (inferred) |
| Masked region modeling (MRM) | Image regions with masked visual features + text tokens | 1D (t) (inferred); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Reconstructed masked regions (features/classes) | 2D (x, y) (inferred) | Capped (inferred) |
| Image-text matching (ITM) | Sentence tokens + image regions | 1D (t) (inferred); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Match/non-match label | 0D (inferred) | Fixed (inferred) |
| Word-region alignment (WRA) | Word tokens + image regions | 1D (t) (inferred); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Word-region transport plan / OT alignment score | 2D (x, y) (inferred); 0D (inferred) | Capped (inferred) |
| Visual question answering (VQA) | Image + natural language question | 2D (x, y) (inferred); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer label | 0D (inferred) | Fixed (inferred) |
| Visual commonsense reasoning (VCR) | Image + question/answer/rationale text choices | 2D (x, y) (inferred); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Selected answer/rationale class | 0D (inferred) | Fixed (inferred) |
| Natural language visual reasoning (NLVR2) | Image pair + natural language statement | 2D (x, y) (inferred); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | True/false label | 0D (inferred) | Fixed (inferred) |
| Visual entailment (SNLI-VE) | Image + natural language statement | 2D (x, y) (inferred); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Entailment/Neutral/Contradiction label | 0D (inferred) | Fixed (inferred) |
| Image-text retrieval | Image-text queries/candidates | 2D (x, y) (inferred); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Ranked retrieval results / similarity ranking | 1D (t) (inferred) | Capped (inferred) |
| Referring expression comprehension | Query phrase/sentence + image region proposals | 2D (x, y) (inferred); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Selected target region proposal | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper covers ten model-handled tasks: four multimodal pre-training tasks (MLM, MRM, ITM, WRA) and six downstream Vision-and-Language tasks (VQA, VCR, NLVR2, Visual Entailment, Image-Text Retrieval, and Referring Expression Comprehension). Inputs are consistently multimodal (text tokens plus image regions/images), which supports 1D (t) and 2D (x, y) task domains. Outputs span 0D labels (classification/matching), 1D ranked lists (retrieval), and 2D region-level outputs/alignment structures. Dynamics/Attention/State labels are inferred as mostly Capped, Static, and Direct from the described fixed-transformer processing over provided image-text inputs.

## Evidence
### Task: Masked language modeling (MLM)
- "Masked Language Modeling (MLM) ... The goal is to predict these masked words based on the observation of their surrounding words  $\mathbf{w}_{\setminus \mathbf{m}}$  and all image regions  $\mathbf{v}$" (Section 3.2, Masked Language Modeling)
- "our MRM and MLM are in analogy to BERT, where we randomly mask some words or regions from the input and learn to recover the words or regions as the output of Transformer" (Section 3.1)
- Inference: `1D (t); 2D (x, y)`, `Capped`, `Static`, and `Direct` are inferred from token-sequence + image-region input and standard Transformer processing over provided inputs; the paper does not use glossary labels explicitly (Section 3.1; Section 3.2).

### Task: Masked region modeling (MRM)
- "Masked Region Modeling (MRM) ... we also sample image regions and mask their visual features with a probability of 15%. The model is trained to reconstruct the masked regions  $\mathbf{v_m}$  given the remaining regions ... and all the words  $\mathbf{w}$" (Section 3.2, Masked Region Modeling)
- "We design four pre-training tasks: Masked Language Modeling (MLM), Masked Region Modeling (MRM, with three variants), Image-Text Matching (ITM), and Word-Region Alignment (WRA)." (Abstract)
- Inference: `2D (x, y)` output plus `Capped/Static/Direct` are inferred from region reconstruction over a finite set of detected regions and Transformer-based processing (Abstract; Section 3.2).

### Task: Image-text matching (ITM)
- "Image-Text Matching (ITM) ... The inputs to ITM are a sentence and a set of image regions, and the output is a binary label  $y \in \{0, 1\}$" (Section 3.2, Image-Text Matching)
- "We also learn an instance-level alignment between the whole image and the sentence via ITM." (Section 3.1)
- Inference: `0D` and `Fixed` output are inferred from the explicit binary label design; `Capped/Static/Direct` are inferred from fixed-form pair scoring over provided sentence-region inputs (Section 3.1; Section 3.2).

### Task: Word-region alignment (WRA)
- "Word-Region Alignment (WRA) We use Optimal Transport (OT) for WRA, where a transport plan  $\mathbf{T} \in \mathbb{R}^{T \times K}$  is learned to optimize the alignment between  $\mathbf{w}$  and  $\mathbf{v}$" (Section 3.2, Word-Region Alignment)
- "we propose WRA via the use of Optimal Transport, which effectively calculates the minimum cost of transporting the contextualized image embeddings to word embeddings" (Section 3.1)
- Inference: Output dimension is marked `2D (x, y) (inferred); 0D (inferred)` because the task yields a token-region transport matrix and OT score/loss; `Capped/Static/Direct` are inferred from finite token/region sets and non-interactive Transformer processing (Section 3.1; Section 3.2).

### Task: Visual question answering (VQA)
- "In VQA, VCR and NLVR<sup>2</sup> tasks, given an input image (or a pair of images) and a natural language question (or description), the model predicts an answer" (Section 4.1)
- "Visual Question Answering (VQA) ... take 3129 most frequent answers as answer candidates ... At inference time, the max-probable answer is selected as the predicted answer." (Appendix A.2, Visual Question Answering)
- Inference: `2D (x, y); 1D (t)` input and `0D` output are inferred from image + tokenized question to one answer label; output dynamics `Fixed` is inferred from fixed candidate answers; `Static/Direct` follow standard encoder-plus-classifier usage (Section 4.1; Appendix A.2).

### Task: Visual commonsense reasoning (VCR)
- "Visual Commonsense Reasoning (VCR) VCR can be decomposed into two multiple-choice sub-tasks: question-answering task  $(Q \to A)$  and answerjustification task  $(QA \to R)$" (Appendix A.2, Visual Commonsense Reasoning)
- "we concatenate the question ... and each answer (rationale) choice from the four possible answer (rationale) candidates ... train a classifier over two classes (''right'' or ''wrong'')" (Appendix A.2, Visual Commonsense Reasoning)
- Inference: `2D (x, y); 1D (t)` and `0D` are inferred from image+text multiple-choice classification; output dynamics `Fixed` is inferred from bounded choices/classes; `Static/Direct` are inferred from feed-forward task formulation (Appendix A.2).

### Task: Natural language visual reasoning (NLVR2)
- "Natural Language for Visual Reasoning for Real (NLVR<sup>2</sup>) ... The goal is to determine whether a natural language statement is true about the given image pair." (Appendix A.2, NLVR2)
- "An MLP transform is applied on the [CLS] output for binary classification ... final ... true/false classification." (Appendix A.2, NLVR2)
- Inference: `2D (x, y); 1D (t)` input is inferred from image-pair + statement; `0D` and `Fixed` output are inferred from binary true/false classification; `Static/Direct` are inferred from fixed-input classification heads (Appendix A.2).

### Task: Visual entailment (SNLI-VE)
- "For Visual Entailment ... The goal is to predict whether a given image semantically entails an input sentence. Classification accuracy over three classes ('Entailment', 'Neutral' and 'Contradiction') is used" (Section 4.1)
- "we treat SNLI-VE as a three-way classification problem" (Appendix A.2, Visual Entailment)
- Inference: `2D (x, y); 1D (t)` input and `0D` output are inferred from image+sentence to one of three labels; `Fixed` output dynamics follows three-way classification; `Static/Direct` are inferred from the described MLP-on-[CLS] setup (Section 4.1; Appendix A.2).

### Task: Image-text retrieval
- "For Image-Text Retrieval, we consider two datasets (COCO and Flickr30K) and evaluate the model in two settings: Image Retrieval (IR) and Text Retrieval (TR)." (Section 4.1)
- "For Image-Text Retrieval, we formulate it as a ranking problem." (Section 4.1)
- Inference: `1D (t)` output is inferred as an ordered ranking over candidates; `Capped` dynamics is inferred from finite candidate pools/negative sampling during finetuning; `Static/Direct` are inferred from similarity scoring over provided pairs (Section 4.1; Appendix A.2).

### Task: Referring expression comprehension
- "Referring Expression (RE) Comprehension requires the model to select the target from a set of image region proposals given the query description." (Section 4.1)
- "To finetune UNITER on this task, we add a MLP layer on top of the region outputs ... to compute the alignment score between the query phrase/sentence and each region." (Appendix A.2, Referring Expression Comprehension)
- Inference: `2D (x, y)` output is inferred as selecting a target region proposal; `Capped` dynamics is inferred from finite proposal sets; `Static/Direct` are inferred from region-wise scoring over provided query+region inputs (Section 4.1; Appendix A.2).
