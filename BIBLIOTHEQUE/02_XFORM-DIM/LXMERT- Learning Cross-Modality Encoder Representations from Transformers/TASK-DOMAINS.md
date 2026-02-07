# LXMERT: Learning Cross-Modality Encoder Representations from Transformers (2019)
Source: LXMERT- Learning Cross-Modality Encoder Representations from Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Masked cross-modality language modeling (predict masked words) | Image; sentence with masked words | 1D (t); 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Masked words (tokens) | 1D (t) (inferred) | Not specified in the paper. |
| Masked object prediction (RoI-feature regression) | Image objects with masked RoI features; sentence | 1D (t); 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Object RoI features (for masked objects) | 2D (x, y) (inferred) | Not specified in the paper. |
| Masked object prediction (detected-label classification) | Image objects with masked RoI features; sentence | 1D (t); 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Object labels (detected labels) | 2D (x, y) (inferred) | Not specified in the paper. |
| Cross-modality matching (image-sentence match classification) | Image; sentence (matched or mismatched) | 1D (t); 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Match/mismatch label | 0D (inferred) | Fixed (inferred) |
| Image question answering (pre-training) | Image; question sentence | 1D (t); 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Answer to image-related question | 0D (inferred) | Fixed (inferred) |
| Visual question answering (VQA) | Image; natural language question | 1D (t); 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Answer to question | 0D (inferred) | Fixed (inferred) |
| Visual question answering (GQA) | Image; natural language question | 1D (t); 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Answer to question | 0D (inferred) | Fixed (inferred) |
| Statement verification (NLVR<sup>2</sup>) | Two related images; natural language statement | 1D (t); 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Label whether the statement correctly describes the two images | 0D (inferred) | Fixed (inferred) |

## Summary
LXMERT covers pre-training tasks for masked language modeling, masked object prediction (feature regression and label classification), image-sentence matching, and image question answering, and it is evaluated on VQA, GQA, and NLVR<sup>2</sup>. Inputs consistently combine language sequences with images represented via detected objects and bounding-box positions, so the task domains span 1D token sequences and 2D image spatial structure (inferred). Outputs range from tokens and object properties to single-label decisions or answers; output dynamics are mostly unspecified except where a fixed answer candidate set is described for image QA. Attention and state dynamics are not explicitly classified in the paper.

## Evidence
### Task: Masked cross-modality language modeling (predict masked words)
- "words are randomly masked with a probability of 0.15 and the model is asked to predict these masked words." (Section 3.1.1)
- "our model takes two inputs: an image and its related sentence (e.g., a caption or a question)." (Section 2 Model Architecture)
- "Each image is represented as a sequence of objects, and each sentence is represented as a sequence of words." (Section 2 Model Architecture)
- "Each object o_j is represented by its position feature (i.e., bounding box coordinates) p_j" (Section 2.1 Input Embeddings)
- Inference: In Dimension and Out Dimension are inferred from the word sequence and image object positions described above.

### Task: Masked object prediction (RoI-feature regression)
- "randomly masking objects (i.e., masking RoI features with zeros)" (Section 3.1.2)
- "RoI-Feature Regression regresses the object RoI feature f_i with L2 loss" (Section 3.1.2)
- "Each image is represented as a sequence of objects, and each sentence is represented as a sequence of words." (Section 2 Model Architecture)
- "Each object o_j is represented by its position feature (i.e., bounding box coordinates) p_j" (Section 2.1 Input Embeddings)
- Inference: In Dimension and Out Dimension are inferred from the word sequence and object bounding-box positions described above.

### Task: Masked object prediction (detected-label classification)
- "randomly masking objects (i.e., masking RoI features with zeros)" (Section 3.1.2)
- "Detected-Label Classification learns the labels of masked objects with cross-entropy loss." (Section 3.1.2)
- "Each image is represented as a sequence of objects, and each sentence is represented as a sequence of words." (Section 2 Model Architecture)
- "Each object o_j is represented by its position feature (i.e., bounding box coordinates) p_j" (Section 2.1 Input Embeddings)
- Inference: In Dimension and Out Dimension are inferred from the word sequence and object bounding-box positions described above.

### Task: Cross-modality matching (image-sentence match classification)
- "we train a classifier to predict whether an image and a sentence match each other." (Section 3.1.3)
- "our model takes two inputs: an image and its related sentence (e.g., a caption or a question)." (Section 2 Model Architecture)
- "Each image is represented as a sequence of objects, and each sentence is represented as a sequence of words." (Section 2 Model Architecture)
- Inference: In Dimension is inferred from the image and word sequence inputs; Out Dimension and Out Dynamics are inferred because the task predicts a single match/mismatch label.

### Task: Image question answering (pre-training)
- "We ask the model to predict the answer to these image-related questions" (Section 3.1.3)
- "we create a joint answer table with 9500 answer candidates" (Section 3.3 Pre-Training Procedure)
- "our model takes two inputs: an image and its related sentence (e.g., a caption or a question)." (Section 2 Model Architecture)
- Inference: In Dimension is inferred from the image and word sequence inputs; Out Dimension and Out Dynamics are inferred from the single-answer prediction with a fixed answer candidate set.

### Task: Visual question answering (VQA)
- "The goal of visual question answering (VQA) (Antol et al., 2015) is to answer a natural language question related to an image." (Appendix A)
- "our model takes two inputs: an image and its related sentence (e.g., a caption or a question)." (Section 2 Model Architecture)
- Inference: In Dimension is inferred from the image and word sequence inputs; Out Dimension and Out Dynamics are inferred because the task predicts a single answer.

### Task: Visual question answering (GQA)
- "The task of GQA (Hudson and Manning, 2019) is same as VQA (i.e., answer single-image related questions)" (Appendix A)
- "our model takes two inputs: an image and its related sentence (e.g., a caption or a question)." (Section 2 Model Architecture)
- Inference: In Dimension is inferred from the image and word sequence inputs; Out Dimension and Out Dynamics are inferred because the task predicts a single answer.

### Task: Statement verification (NLVR<sup>2</sup>)
- "Each datum in NLVR<sup>2</sup> contains two related natural images and one natural language statement." (Appendix A)
- "The task is to predict whether the statement correctly describes these two images or not." (Appendix A)
- Inference: In Dimension is inferred from the image and language statement inputs; Out Dimension and Out Dynamics are inferred because the task predicts a single label.
