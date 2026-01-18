## 1. Basic Metadata

- Title: "VISUALBERT: A SIMPLE AND PERFORMANT BASELINE FOR VISION AND LANGUAGE" (Title)
- Authors: "Liunian Harold Li<sup>†</sup>, Mark Yatskar\*, Da Yin°, Cho-Jui Hsieh<sup>†</sup> & Kai-Wei Chang<sup>†</sup>" (Title)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper proposes VisualBERT as "a simple and flexible framework for modeling a broad range of vision-and-language tasks" and introduces "two visually-grounded language model objectives for pre-training VisualBERT on image caption data" to improve downstream task performance (ABSTRACT).

## 3. Tasks Evaluated

- Task name: Visual Question Answering (VQA 2.0)
  - Task type: Classification
  - Dataset(s) used: VQA 2.0; COCO images
  - Domain: natural images (COCO)
  - Quotes: "Given an image and a question, the task is to correctly answer the question." (Section 4.1 VQA)
  - Quotes: "We use the VQA 2.0 (Goyal et al., 2017), consisting of over 1 million questions about images from COCO." (Section 4.1 VQA)
  - Quotes: "we follow the processing procedure of Pythia and consider it a classification problem, where the model only needs to choose one answer from a limited answer pool." (Appendix A VQA)

- Task name: Visual Commonsense Reasoning (VCR)
  - Task type: Classification; Reasoning / relational
  - Dataset(s) used: VCR
  - Domain: movie scenes
  - Quotes: "VCR consists of 290k questions derived from 110k movie scenes, where the questions focus on visual commonsense." (Section 4.2 VCR)
  - Quotes: "The task is decomposed into two multi-choice sub-tasks wherein we train individual models: question answering (Q  $\rightarrow$  A) and answer justification (QA  $\rightarrow$  R)." (Section 4.2 VCR)

- Task name: Natural Language for Visual Reasoning (NLVR $^2$)
  - Task type: Classification; Reasoning / relational
  - Dataset(s) used: NLVR $^2$
  - Domain: web images
  - Quotes: "The task is to determine whether a natural language caption is true about a pair of images." (Section 4.3 NLVR$^2$)
  - Quotes: "The dataset consists of over 100k examples of English sentences paired with web images." (Section 4.3 NLVR$^2$)

- Task name: Region-to-Phrase Grounding (Flickr30K Entities)
  - Task type: Detection
  - Dataset(s) used: Flickr30K Entities
  - Domain: natural images
  - Quotes: "Flickr30K Entities dataset tests the ability of systems to ground phrases in captions to bounding regions in the image." (Section 4.4 FLICKR30K ENTITIES)
  - Quotes: "The task is, given spans from a sentence, selecting the bounding regions they correspond to." (Section 4.4 FLICKR30K ENTITIES)
  - Quotes: "The dataset consists of 30k images and nearly 250k annotations." (Section 4.4 FLICKR30K ENTITIES)

## 4. Domain and Modality Scope

- Evaluation performed on multiple modalities (vision and language): "We propose VisualBERT, a simple and flexible framework for modeling a broad range of vision-and-language tasks." (ABSTRACT)
- Evaluation performed on multiple domains within the same modality (images): "VCR consists of 290k questions derived from 110k movie scenes" and "The dataset consists of over 100k examples of English sentences paired with web images." (Section 4.2 VCR; Section 4.3 NLVR$^2$)
- Domain generalization or cross-domain transfer claim: Not claimed explicitly; the paper notes cross-domain differences: "Despite substantial domain difference between COCO and VCR, with VCR covering scenes from movies, pre-training on COCO still helps significantly." (Section 4.2 VCR)

## 5. Model Sharing Across Tasks

Evidence for overall training scheme: "It is pre-trained with a masked language modeling (Objective 1), and sentence-image prediction task (Objective 2), on caption data and then fine-tuned for different tasks." (Figure 2 caption)

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| VQA 2.0 | No (separate fine-tuning per task) | Yes | Yes | "Fine-Tuning This step mirrors BERT fine-tuning, where a task-specific input, output, and objective are introduced" (Section 3.3 TRAINING VISUALBERT); "the representation of the [MASK] token is fed into an output layer for classification" (Appendix A VQA) |
| VCR | No (separate fine-tuning per task) | Yes | Yes | "Fine-Tuning This step mirrors BERT fine-tuning, where a task-specific input, output, and objective are introduced" (Section 3.3 TRAINING VISUALBERT); "The model is trained to classify which of the four input sequences is correct." (Appendix B VCR) |
| NLVR $^2$ | No (separate fine-tuning per task) | Yes | Yes | "Fine-Tuning This step mirrors BERT fine-tuning, where a task-specific input, output, and objective are introduced" (Section 3.3 TRAINING VISUALBERT); "an auxiliary task is added to decide whether the caption in an training example is true." (Appendix C NLVR$^2$) |
| Flickr30K Entities | No (separate fine-tuning per task) | Yes | Yes | "Fine-Tuning This step mirrors BERT fine-tuning, where a task-specific input, output, and objective are introduced" (Section 3.3 TRAINING VISUALBERT); "we introduce an additional self-attention block and use the average attention weights from each head to predict the alignment between boxes and phrases." (Section 4.4 FLICKR30K ENTITIES) |

## 6. Input and Representation Constraints

- Unordered region tokens: "image features extracted from object proposals are treated as unordered input tokens and fed into VisualBERT along with text." (Section 1 Introduction)
- Region-level inputs from detectors: "Each  $f \in F$  corresponds to a bounding region in the image, derived from an object detector." (Section 3.2 VISUALBERT)
- Visual embedding components (fixed dimensionality alignment): "Each embedding in F is computed by summing three embeddings: (1)  $f_o$ , a visual feature representation of the bounding region of f, computed by a convolutional neural network, (2)  $f_s$ , a segment embedding indicating it is an image embedding as opposed to a text embedding, and (3)  $f_p$ , a position embedding" (Section 3.2 VISUALBERT) and "If text and visual input embeddings are of different dimension, we project the visual embeddings into a space of the same dimension as the text embeddings." (Section 4 EXPERIMENT)
- Fixed text length cap: "text sequences whose lengths are longer than 128 are capped." (Section 4 EXPERIMENT)
- Fixed number of region proposals (task-specific): "We use an off-the-shelf detector from Detectron (Girshick et al., 2018) to provide image features and use 144 proposals per image." (Section 4.3 NLVR$^2$)
- Fixed number of region features in analysis: "all these models are trained with only 36 features per image (including the full model)." (Section 5.1 ABLATION STUDY)
- Avoiding grid-level features due to sequence length: "We do not use grid-level features from ResNet152 because it results in longer sequences and longer training time." (Appendix A VQA)
- Gold boxes in VCR: 'Image features are obtained from a ResNet50 (He et al., 2016) and "gold" detection bounding boxes and segmentations provided in the dataset are used<sup>3</sup>.' (Section 4.2 VCR)
- Fixed or variable input resolution: Not specified.
- Fixed patch size: Not specified.
- Fixed number of tokens (beyond the text cap and proposal counts above): Not specified.
- Padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: "text sequences whose lengths are longer than 128 are capped." (Section 4 EXPERIMENT)
- Sequence length fixed or variable: Variable, with a cap at 128 for text sequences (Section 4 EXPERIMENT).
- Attention type: Global self-attention via Transformer: "VisualBERT consists of a stack of Transformer layers that implicitly align elements of an input text and regions in an associated input image with self-attention." (ABSTRACT)
- Mechanisms to manage computational cost: "text sequences whose lengths are longer than 128 are capped" (Section 4 EXPERIMENT) and "We do not use grid-level features from ResNet152 because it results in longer sequences and longer training time." (Appendix A VQA)

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: Position embeddings added to token embeddings: "Each embedding  $e \in E$  is computed as the sum of 1) a token embedding  $e_t$ , specific to the subword, 2) a segment embedding  $e_s$ , indicating which part of text the token comes from ... and 3) a position embedding  $e_p$ , indicating the position of the token in the sentence." (Section 3.1 BACKGROUND)
- Visual position embeddings when alignments exist: "(3)  $f_p$ , a position embedding, which is used when alignments between words and bounding regions are provided as part of the input, and set to the sum of the position embeddings corresponding to the aligned words (see VCR in  $\S4$ )." (Section 3.2 VISUALBERT)
- Where applied: Input embeddings are sums of token/segment/position embeddings (Section 3.1 BACKGROUND; Section 3.2 VISUALBERT).
- Fixed across all experiments vs modified per task: Not explicitly stated; a task-specific use is described when alignments are provided (VCR) (Section 3.2 VISUALBERT).
- Ablated or compared against alternatives: Not reported.

## 9. Positional Encoding as a Variable

- Treated as a core research variable or fixed assumption: Fixed architectural assumption described as part of the input embedding sums (Section 3.1 BACKGROUND; Section 3.2 VISUALBERT).
- Multiple positional encodings compared: Not reported.
- Claims that PE choice is not critical or secondary: Not claimed.

## 10. Evidence of Constraint Masking

- Model size: "The Transformer encoder in all models has the same configuration as BERT<sub>BASE</sub>: 12 layers, a hidden size of 768, and 12 self-attention heads." (Section 4 EXPERIMENT)
- Dataset sizes: "COCO ... has around 100k images with 5 captions each." (Section 4 EXPERIMENT); "consisting of over 1 million questions about images from COCO." (Section 4.1 VQA); "VCR consists of 290k questions derived from 110k movie scenes" (Section 4.2 VCR); "The dataset consists of over 100k examples of English sentences paired with web images." (Section 4.3 NLVR$^2$); "The dataset consists of 30k images and nearly 250k annotations." (Section 4.4 FLICKR30K ENTITIES)
- Performance gains attributed to scaling data or architecture: "Results confirm that task-agnostic pretraining (C1) and early fusion of vision and language (C2) are essential for VisualBERT." (Section 5.1 ABLATION STUDY) and "Despite substantial domain difference between COCO and VCR... pre-training on COCO still helps significantly." (Section 4.2 VCR)
- Performance gains attributed to scaling model size: Not stated.
- Training tricks: Not emphasized beyond standard optimization and pre-training procedures (not attributed as primary gains).

## 11. Architectural Workarounds

- Object-proposal tokens for images: "image features extracted from object proposals are treated as unordered input tokens" (Section 1 Introduction) to allow Transformer processing without dense grids.
- Segment embeddings to distinguish modalities: "a segment embedding indicating it is an image embedding as opposed to a text embedding" (Section 3.2 VISUALBERT).
- Segment embeddings to separate paired images (NLVR$^2$): "assign features from different images with different segment embeddings." (Section 4.3 NLVR$^2$)
- Alignment-aware position embeddings (VCR): "The dataset also provides alignments between words and bounding regions that are referenced to in the text, which we utilize by using the same position embeddings for matched words and regions." (Section 4.2 VCR)
- Task-specific alignment head for Flickr30K: "we introduce an additional self-attention block and use the average attention weights from each head to predict the alignment between boxes and phrases." (Section 4.4 FLICKR30K ENTITIES)

## 12. Explicit Limitations and Non-Claims

- Future work: "For future work, we are curious about whether we could extend VisualBERT to image-only tasks, such as scene graph parsing and situation recognition. Pre-training VisualBERT on larger caption datasets such as Visual Genome and Conceptual Caption is also a valid direction." (Section 6 CONCLUSION AND FUTURE WORK)
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: Multiple image domains (COCO, movie scenes, web images) but all within vision-and-language datasets.
- Task structure: Four specified tasks (VQA, VCR, NLVR$^2$, Flickr30K grounding) with fixed evaluation protocols.
- Representation rigidity: Region proposals as unordered tokens, text length capped at 128, fixed proposal counts per dataset.
- Model sharing vs specialization: Shared pretraining, then task-specific pretraining and fine-tuning with task-specific heads.
- Role of positional encoding: Standard BERT-style position embeddings with alignment-based visual positions; not varied across experiments.

### 14. Final Classification

**Multi-task, multi-domain (constrained).** The paper evaluates "four different types of vision-and-language applications" (Section 4 EXPERIMENT), spanning datasets from "images from COCO" (Section 4.1 VQA), "movie scenes" (Section 4.2 VCR), and "web images" (Section 4.3 NLVR$^2$), indicating multiple domains within the same modality. The setup remains constrained because models are "fine-tuned for different tasks" with "a task-specific input, output, and objective" (Figure 2 caption; Section 3.3 TRAINING VISUALBERT), rather than a single unrestrained multi-task model.
