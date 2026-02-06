# EVA02-AT: Egocentric Video-Language Understanding with Spatial-Temporal Rotary Positional Embeddings and Symmetric Optimization (Not specified in the paper.)
Source: EVA02-AT- Egocentric Video-Language with Spatial-Temporal RoPE.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Video-text retrieval (multi-instance retrieval) | video clips; text narrations | 3D (x, y, t) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | pairwise match labels / similarity scores between video and text | 2D (x, y) (inferred) | Not specified in the paper. |
| Action recognition (video-to-text) | video clips | 3D (x, y, t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | action labels | 0D (inferred) | Not specified in the paper. |
| Multiple-choice question answering (EgoMCQ) | video clips (inferred); multiple-choice questions (text) | 3D (x, y, t) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | answer choice (inferred) | 0D (inferred) | Not specified in the paper. |

## Summary
EVA02-AT is evaluated on egocentric video-language tasks spanning multi-instance video-text retrieval, video-to-text action recognition, and EgoMCQ multiple-choice QA. Inputs include spatiotemporal video and text (narrations or questions), while outputs are pairwise match labels or discrete action/answer labels; dimensions therefore span 3D (x, y, t) and 1D (t) with 0D/2D outputs (inferred). The architecture uses global joint attention, implying static attention with constructed internal representations, while input/output dynamics are largely not specified in the paper (inferred).

## Evidence
### Task: Video-text retrieval (multi-instance retrieval)
- "the objective of the video text retrieval task is to learn a similarity calculation function  $S(\cdot)$  that satisfies  $S(\mathcal{V}, \mathcal{T}) = C$ ." (Section III Preliminary, Learning objective)
- "we directly evaluate the pretrained model on EK-100's multi-instance retrieval (MIR) challenge" (Section V.A Datasets and Implementation Details)
- Inference: In/Out dimensions and attention/state dynamics are inferred from "an input video sequence  $\mathbf{v} \in \mathbb{R}^{C \times T \times H \times W}$" and "the attention score between query and key becomes a global attention among all the patches in the video clip," plus "We introduce two distinct learnable positional embeddings." (Section IV.A EVA-02 AT Transformer; Joint Spatial-Temporal Attention)

### Task: Action recognition (video-to-text)
- "CharadesEgo Video-to-Text action recognition task." (Section V.B Compare with State-of-the-Arts)
- "The Charades-Ego dataset only contains hard labels, but there could be multiple different hard labels for each video clip." (Section V.A Datasets and Implementation Details)
- Inference: Input dimension and attention/state dynamics are inferred from "an input video sequence  $\mathbf{v} \in \mathbb{R}^{C \times T \times H \times W}$" and "the attention score between query and key becomes a global attention among all the patches in the video clip," plus "We introduce two distinct learnable positional embeddings." (Section IV.A EVA-02 AT Transformer; Joint Spatial-Temporal Attention)

### Task: Multiple-choice question answering (EgoMCQ)
- "After pretraining, we evaluate models on the Ego4D Multiple-Choice Questions (EgoMCQ) benchmark." (Section V.A Datasets and Implementation Details)
- "EgoMCQ. We directly evaluate the EgoMCQ performance after pretraining the model on the Ego4D dataset." (Section V.B Compare with State-of-the-Arts)
- Inference: The video-language nature and question format are inferred from "Egocentric video-language understanding" and the "Multiple-Choice Questions" benchmark wording, with dimensions/attention/state inferred from "an input video sequence  $\mathbf{v} \in \mathbb{R}^{C \times T \times H \times W}$" and "the attention score between query and key becomes a global attention among all the patches in the video clip," plus "We introduce two distinct learnable positional embeddings." (Abstract; Section IV.A EVA-02 AT Transformer; Joint Spatial-Temporal Attention)
