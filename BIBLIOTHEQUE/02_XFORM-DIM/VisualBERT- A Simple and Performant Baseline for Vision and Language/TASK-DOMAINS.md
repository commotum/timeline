# VISUALBERT: A SIMPLE AND PERFORMANT BASELINE FOR VISION AND LANGUAGE (Not specified in the paper.)
Source: VisualBERT- A Simple and Performant Baseline for Vision and Language.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Masked language modeling with image context | Masked text tokens; image region embeddings | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Masked token identities | 1D (t) (inferred) | Capped (inferred) |
| Sentence-image match classification | Two-caption text segment; image region embeddings | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Match/non-match label | 0D (inferred) | Fixed (inferred) |
| Visual question answering classification (VQA 2.0) | Question tokens; image regions | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Answer label from limited answer pool | 0D (inferred) | Fixed (inferred) |
| Visual commonsense question answering classification (VCR Q -> A) | Question tokens; answer-choice tokens; image regions | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Correct answer-choice label (1 of 4) | 0D (inferred) | Fixed (inferred) |
| Visual commonsense answer-justification classification (VCR QA -> R) | Question+answer tokens; rationale-choice tokens; image regions | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Correct rationale-choice label (1 of 4) | 0D (inferred) | Fixed (inferred) |
| Caption truth classification over image pairs (NLVR2) | Caption tokens; paired image regions | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | True/false label | 0D (inferred) | Fixed (inferred) |
| Region-to-phrase grounding (Flickr30K Entities) | Phrase spans in caption; image regions/boxes | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Selected bounding region per phrase | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper covers two pre-training tasks plus four downstream vision-language applications (with VCR split into two explicitly trained sub-tasks). Across rows, inputs are multimodal text-plus-image-region structures, yielding a combined 1D (t) and 2D (x, y) input domain. Output domains are mostly 0D classification labels, except Flickr30K grounding which outputs image-region selections in 2D (x, y). Input/output dynamics are capped or fixed, and attention/state are static and constructed (inferred from the fixed-token interface and Transformer joint-representation design).

## Evidence
### Task: Masked language modeling with image context
- "Task-Agnostic Pre-Training Here we train VisualBERT on COCO using two visually-grounded language model objectives. (1) Masked language modeling with the image. Some elements of text input are masked and must be predicted but vectors corresponding to image regions are not masked." (Section 3.3 TRAINING VISUALBERT)
- "Image regions and language are combined with a Transformer to allow the self-attention to discover implicit alignments between language and vision. It is pre-trained with a masked language modeling (Objective 1), and sentence-image prediction task (Objective 2), on caption data and then fine-tuned for different tasks." (Figure 2 caption)
- Inference: In Dimension, In Dynamics, Attention Dynamic, State Dynamic, Out Dimension, and Out Dynamics are inferred from multimodal tokenized input plus bounded sequence length and Transformer processing: "image features extracted from object proposals are treated as unordered input tokens and fed into VisualBERT along with text" (Section 1 Introduction), "text sequences whose lengths are longer than 128 are capped" (Section 4 EXPERIMENT), and "the input embeddings E are then passed through a multi-layer Transformer that builds up a contextualized representation" (Section 3.1 BACKGROUND).

### Task: Sentence-image match classification
- "(2) Sentence-image prediction. For COCO, where there are multiple captions corresponding to one image, we provide a text segment consisting of two captions." (Section 3.3 TRAINING VISUALBERT)
- "The model is trained to distinguish these two situations." (Section 3.3 TRAINING VISUALBERT)
- Inference: In Dimension, In Dynamics, Attention Dynamic, State Dynamic, Out Dimension, and Out Dynamics are inferred from the same multimodal bounded-input Transformer setup: "VisualBERT consists of a stack of Transformer layers that implicitly align elements of an input text and regions in an associated input image with self-attention" (ABSTRACT) and "text sequences whose lengths are longer than 128 are capped" (Section 4 EXPERIMENT).

### Task: Visual question answering classification (VQA 2.0)
- "Given an image and a question, the task is to correctly answer the question." (Section 4.1 VQA)
- "Though the answers of VQA are open-ended, we follow the processing procedure of Pythia and consider it a classification problem, where the model only needs to choose one answer from a limited answer pool." (Appendix A VQA)
- Inference: In Dimension, In Dynamics, Attention Dynamic, State Dynamic, and Out Dimension are inferred from text+image region inputs and Transformer processing: "image features extracted from object proposals are treated as unordered input tokens and fed into VisualBERT along with text" (Section 1 Introduction). Out Dynamics = Fixed is inferred from "We train the model to predict the 3,129 most frequent answers" (Section 4.1 VQA).

### Task: Visual commonsense question answering classification (VCR Q -> A)
- "The task is decomposed into two multi-choice sub-tasks wherein we train individual models: question answering (Q  ->  A) and answer justification (QA  ->  R)." (Section 4.2 VCR)
- "For each sub-task, each training example contains four choices and we construct four input sequences, each containing the concatenation of the given question, a choice, and an image. ... The model is trained to classify which of the four input sequences is correct." (Appendix B VCR)
- Inference: In Dimension, In Dynamics, Attention Dynamic, State Dynamic, and Out Dimension are inferred from text-sequence plus image-region input processed by Transformer self-attention (Section 3.2 VISUALBERT; ABSTRACT). Out Dynamics = Fixed is inferred from the explicit four-choice classification interface (Appendix B VCR).

### Task: Visual commonsense answer-justification classification (VCR QA -> R)
- "The task is decomposed into two multi-choice sub-tasks wherein we train individual models: question answering (Q  ->  A) and answer justification (QA  ->  R)." (Section 4.2 VCR)
- "When the model performs QA  ->  R, the \"question\" part contains the original question and the correct choice, and the \"choice\" is a possible rationale." (Appendix B VCR)
- Inference: In Dimension, In Dynamics, Attention Dynamic, State Dynamic, and Out Dimension are inferred from text+image region inputs and the shared VisualBERT Transformer mechanism (Section 3.2 VISUALBERT; ABSTRACT). Out Dynamics = Fixed is inferred from "The model is trained to classify which of the four input sequences is correct" (Appendix B VCR).

### Task: Caption truth classification over image pairs (NLVR2)
- "The task is to determine whether a natural language caption is true about a pair of images." (Section 4.3 NLVR2)
- "For each training example in NLVR2, we construct a sequence consisting of the caption and image features from two images." (Appendix C NLVR2)
- Inference: In Dimension, In Dynamics, Attention Dynamic, State Dynamic, Out Dimension, and Out Dynamics are inferred from paired images plus caption tokens as bounded multimodal input and boolean truth decision: "assign features from different images with different segment embeddings" and "use 144 proposals per image" (Section 4.3 NLVR2), plus the caption-truth decision statement (Section 4.3 NLVR2).

### Task: Region-to-phrase grounding (Flickr30K Entities)
- "Flickr30K Entities dataset tests the ability of systems to ground phrases in captions to bounding regions in the image. The task is, given spans from a sentence, selecting the bounding regions they correspond to." (Section 4.4 FLICKR30K ENTITIES)
- "For a phrase to be grounded, we take whichever box receives the most attention from the last sub-word of the phrase as the model prediction." (Section 4.4 FLICKR30K ENTITIES)
- Inference: In Dimension, In Dynamics, Attention Dynamic, State Dynamic, Out Dimension, and Out Dynamics are inferred from phrase-span text plus image boxes as input and box-selection output over detector-provided regions; this is supported by "image features from a Faster R-CNN pre-trained on Visual Genome are used" (Section 4.4 FLICKR30K ENTITIES) and the fixed-sequence constraint "text sequences whose lengths are longer than 128 are capped" (Section 4 EXPERIMENT).
