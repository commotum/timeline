# LLaVA-4D: Embedding SpatioTemporal Prompt into LMMs for 4D Scene Understanding (Not specified in the paper.)
Source: LLaVA-4D- Embedding Spatiotemporal Prompt into LMMs.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dense captioning | multi-view video/image inputs; text instructions | 2D (x, y); 3D (x, y, z) or (x, y, t); 4D (x, y, z, t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text captions / responses | 1D (t) (inferred) | Not specified in the paper. |
| Visual question answering (QA) | multi-view video/image inputs; text questions/instructions | 2D (x, y); 3D (x, y, z) or (x, y, t); 4D (x, y, z, t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text answers / responses | 1D (t) (inferred) | Not specified in the paper. |
| Visual grounding (VG) | multi-view video/image inputs; text referring expressions/instructions | 2D (x, y); 3D (x, y, z) or (x, y, t); 4D (x, y, z, t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | grounded object localization (spatial/temporal) | 2D (x, y); 3D (x, y, z) or (x, y, t); 4D (x, y, z, t) (inferred) | Not specified in the paper. |
| Spatiotemporal segmentation (semantic/action masks) | multi-view video input sequence; text instruction | 4D (x, y, z, t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | semantic/action segmentation masks (inferred) | 3D (x, y, z) or (x, y, t) (inferred) | Not specified in the paper. |

## Summary
The paper covers dense captioning, visual QA, and visual grounding across 2D/3D/4D vision-language data, and also demonstrates extension to spatiotemporal segmentation via semantic/action masks. Inputs are described as multi-view video sequences (and image-format inputs) paired with text instructions, with text responses for captioning/QA and spatial-temporal localization for grounding. The work explicitly constructs 4D coordinates [x, y, z, t], but does not specify interface dynamics, attention policy, or state construction.

## Evidence
### Task: Dense captioning
- "These datasets cover dense captioning (DC), visual QA and visual grounding (VG) tasks with a total of 654.5K samples." (Section 4.1 Our Chat4D Dataset)
- "Chat4D dataset includes 2D, 3D, and 4D vision-language training sets for dense captioning, QA, and visual grounding." (Figure 4 caption)
- "We evaluate the quality of generated text response for Scan2Cap and ScanQA in terms of CiDEr (C), BLEU-4 (B-4), METEOR (M)." (Section 5.1 Evaluation Metric)
- Inference: Out Dimension set to "1D (t)" because the paper evaluates a "generated text response" for captioning tasks. (Section 5.1 Evaluation Metric)

### Task: Visual question answering (QA)
- "These datasets cover dense captioning (DC), visual QA and visual grounding (VG) tasks with a total of 654.5K samples." (Section 4.1 Our Chat4D Dataset)
- "Chat4D dataset includes 2D, 3D, and 4D vision-language training sets for dense captioning, QA, and visual grounding." (Figure 4 caption)
- "We evaluate the quality of generated text response for Scan2Cap and ScanQA in terms of CiDEr (C), BLEU-4 (B-4), METEOR (M)." (Section 5.1 Evaluation Metric)
- Inference: Out Dimension set to "1D (t)" because the paper evaluates a "generated text response" for QA tasks. (Section 5.1 Evaluation Metric)

### Task: Visual grounding (VG)
- "These datasets cover dense captioning (DC), visual QA and visual grounding (VG) tasks with a total of 654.5K samples." (Section 4.1 Our Chat4D Dataset)
- "Chat4D dataset includes 2D, 3D, and 4D vision-language training sets for dense captioning, QA, and visual grounding." (Figure 4 caption)
- "We choose the F1 metric of object prediction precision for Multi3DRefer," (Section 5.1 Evaluation Metric)
- "the accuracy of intersection over unions for grounding task from ScanRef." (Section 5.1 Evaluation Metric)
- "grounding accuracy is divided into spatial and temporal components: S/TAcc." (Section 5.1 Evaluation Metric)
- Inference: Out Dimension set to spatial/temporal localization because grounding is evaluated with object prediction and spatial/temporal accuracy metrics. (Section 5.1 Evaluation Metric)

### Task: Spatiotemporal segmentation (semantic/action masks)
- "we introduce additional spatial semantic and temporal action masks as prompts based on the encoded 4D coordinates to train our model." (Extensibility of Spatiotemporal Prompt)
- "Figure 7: Visualization of spatiotemporal prompt extended to other spatiotemporal vision tasks." (Figure 7 caption)
- "Please segment this dance move." (Figure 7)
- "At 1.4s, [SEG]. At 1.9s, [SEG]. At 2.4s, [SEG]." (Figure 7)
- Inference: Output treated as segmentation masks over time because the paper mentions semantic/action masks and shows [SEG] outputs at timestamps. (Extensibility of Spatiotemporal Prompt; Figure 7)
