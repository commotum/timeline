# ECHOMOTION: UNIFIED HUMAN VIDEO AND MOTION GENERATION VIA DUAL-MODALITY DIFFUSION TRANSFORMER (Year not specified in the paper)
Source: EchoMotion- Unified Human Video and Motion Generation via Dual-Modality Diffusion Transformer.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Text-to-video generation | text prompt | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | video sequence | 3D (x, y, t) (inferred) | Not specified in the paper. |
| Joint video-and-motion generation | text prompt | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | video sequence; SMPL motion sequence | 3D (x, y, t) (inferred); 4D (x, y, z, t) (inferred) | Not specified in the paper. |
| Motion-to-video generation | motion sequence; text prompt | 4D (x, y, z, t) (inferred); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | video sequence | 3D (x, y, t) (inferred) | Not specified in the paper. |
| Video-to-motion generation | video sequence | 3D (x, y, t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | SMPL motion sequence | 4D (x, y, z, t) (inferred) | Not specified in the paper. |
| Text-to-motion generation | text prompt | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | motion parameters | 4D (x, y, z, t) (inferred) | Not specified in the paper. |

## Summary
EchoMotion covers text-conditioned video generation, joint text-to-video-and-motion generation, and cross-modal completion (motion-to-video and video-to-motion), and it is evaluated for motion synthesis from text prompts. Inputs span text prompts, motion sequences, and video sequences, while outputs include video (3D (x, y, t)) and motion parameters/sequences (4D (x, y, z, t)), with these dimensional labels inferred from the modalities described. The paper does not explicitly specify interface dynamics or attention/state dynamics for these tasks.

## Evidence
### Task: Text-to-video generation
- "#### 4.2 TEXT TO VIDEO GENERATION" (Section 4.2)
- "Figure 6: Text-to-video results from EchoMotion, demonstrating both strong prompt alignment and high kinematic plausibility across a diverse range of human-centric scenarios." (Figure 6 caption, Section 4.2)
- Inference: Input/output dimensions inferred from the modality labels in the quotes (text prompts as 1D (t); video sequences as 3D (x, y, t)).

### Task: Joint video-and-motion generation
- "This paper introduces EchoMotion, a system designed to generate videos with corresponding motion sequences from an input text prompt." (Section 3)
- "1) Joint training: generate both video and motion sequences concurrently." (Section 3.2)
- "EchoMotion jointly generates an SMPL motion sequence (left) and video (right), demonstrating a learned joint distribution." (Figure 7 caption, Section 4.2)
- Inference: Input/output dimensions inferred from the modality labels in the quotes (text prompts as 1D (t); video sequences as 3D (x, y, t); SMPL motion sequences as 4D (x, y, z, t)).

### Task: Motion-to-video generation
- "2) Motion-to-video training: motion sequences serve as the conditioning input for video generation." (Section 3.2)
- "Figure 9: Cross-modal completion by EchoMotion. (a) Motion-to-Video synthesis from motion and text." (Figure 9 caption, Section 4.3)
- "it can synthesize a high-fidelity video that precisely follows a given motion sequence (motion-to-video)." (Section 4.3)
- Inference: Input/output dimensions inferred from the modality labels in the quotes (motion sequences as 4D (x, y, z, t); text prompts as 1D (t); video sequences as 3D (x, y, t)).

### Task: Video-to-motion generation
- "3) Video-to-motion training: video sequences are used to condition motion generation." (Section 3.2)
- "it can recover the underlying SMPL motion from an input video (video-to-motion)." (Section 4.3)
- Inference: Input/output dimensions inferred from the modality labels in the quotes (video sequences as 3D (x, y, t); SMPL motion sequences as 4D (x, y, z, t)).

### Task: Text-to-motion generation
- "While EchoMotion is primarily designed for video generation, we also evaluate its motion synthesis quality." (Section A.7.4)
- "we generated a set of 50 diverse prompts using an LLM (Team et al., 2024)" (Section A.7.4)
- "rendered the generated motion parameters into mesh videos" (Section A.7.4)
- Inference: Input/output dimensions inferred from the modality labels in the quotes (text prompts as 1D (t); motion parameters as 4D (x, y, z, t)).
