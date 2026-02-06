# FLASHATTENTION-2: Faster Attention with Better Parallelism and Work Partitioning (2023)
Source: FlashAttention-2- Faster Attention with Better Parallelism and Work Partitioning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Auto-regressive language modeling | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| High-resolution image understanding | images (inferred) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Code generation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | code tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Audio generation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | audio waveforms (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Video generation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | video frames (inferred) | 3D (x, y, t) (inferred) | Not specified in the paper. |
| Long document querying | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Story writing | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper frames FlashAttention-2 as enabling or improving Transformer-based tasks including auto-regressive language modeling, high-resolution image understanding, code/audio/video generation, long document querying, and story writing. Most task inputs, outputs, and dynamics are not specified beyond these mentions; where noted, I inferred modality-typical dimensions (e.g., text/code as 1D sequences, images as 2D grids, video as 3D spatiotemporal). For auto-regressive language modeling, the discussion of sequence length limits and causal masking supports a capped context length and static attention pattern (inferred), while state dynamics remain otherwise unspecified.

## Evidence
### Task: Auto-regressive language modeling
- "One common use case of attention is in auto-regressive language modeling" (Section 3.1, Causal masking)
- "the standard 2k sequence length limit" (Section 1 Introduction)
- "Given input sequences  $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$  where N is the sequence length" (Section 2.2)
- Inference: Treated language modeling as text-token sequences with 1D (t) structure and capped context length, and set attention to Static and state to Direct based on the explicit autoregressive use case, sequence-length limits, and fixed causal masking. (Supported by the quotes above.)

### Task: High-resolution image understanding
- "language modeling and high-resolution image understanding" (Abstract)
- "high resolution images" (Section 1 Introduction)
- Inference: Interpreted the input as images with 2D (x, y) structure based on the explicit references to images. (Supported by the quotes above.)

### Task: Code generation
- "code, audio, and video generation" (Abstract)
- Inference: Treated the output as code tokens with 1D (t) structure because the task is explicitly code generation. (Supported by the quote above.)

### Task: Audio generation
- "code, audio, and video generation" (Abstract)
- Inference: Treated the output as audio waveforms with 1D (t) structure because the task is explicitly audio generation. (Supported by the quote above.)

### Task: Video generation
- "code, audio, and video generation" (Abstract)
- "long-form videos" (Section 1 Introduction)
- Inference: Treated the output as video frames with 3D (x, y, t) structure because the task is explicitly video generation and long-form videos are mentioned. (Supported by the quotes above.)

### Task: Long document querying
- "long document querying and story writing" (Section 1 Introduction)

### Task: Story writing
- "long document querying and story writing" (Section 1 Introduction)
- Inference: Treated the output as text tokens with 1D (t) structure because story writing implies generated text. (Supported by the quote above.)

## CSV Output (required)
task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic
Auto-regressive language modeling,text tokens (inferred),1D (t) (inferred),Capped (inferred),Static (inferred),Direct (inferred),text tokens (inferred),1D (t) (inferred),Capped (inferred)
High-resolution image understanding,images (inferred),"2D (x, y) (inferred)",Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.
Code generation,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,code tokens (inferred),1D (t) (inferred),Not specified in the paper.
Audio generation,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,audio waveforms (inferred),1D (t) (inferred),Not specified in the paper.
Video generation,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,video frames (inferred),"3D (x, y, t) (inferred)",Not specified in the paper.
Long document querying,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.
Story writing,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,text tokens (inferred),1D (t) (inferred),Not specified in the paper.
