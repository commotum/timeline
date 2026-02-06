# Generating Long Sequences with Sparse Transformers (Not specified in the paper.)
Source: Generating Long Sequences with Sparse Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image density modeling (autoregressive generation) | image bytes (pixels) | 3D (x, y, z) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | image bytes (next-token values) | 3D (x, y, z) (inferred) | Fixed (inferred) |
| Text density modeling (autoregressive generation) | text bytes/tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | text bytes/tokens (next-token values) | 1D (t) (inferred) | Capped (inferred) |
| Raw audio density modeling (autoregressive generation) | raw audio bytes | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | raw audio bytes (next-token values) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper frames the task as autoregressive sequence generation/density modeling and evaluates it on images, text, and raw audio. Images are modeled as byte sequences with row/column/channel embeddings, supporting a 3D (x, y, z) domain and fixed-size inputs per dataset, while text and audio are modeled as 1D token sequences with capped context lengths. The attention patterns are predetermined sparse patterns and the system is described as next-token prediction, so attention is static and state is direct (both inferred from the architecture description).

## Evidence
### Task: Image density modeling (autoregressive generation)
- "We empirically test our architecture on density modeling tasks including natural images, text, and raw audio." (Section 7. Experiments)
- "We train strided Sparse Transformers on CIFAR-10 images represented as sequences of 3072 bytes." (Section 7.1. CIFAR-10)
- "For images, we used data embeddings, where  $d_{data}=3$  for the row, column, and channel location of each input byte." (Section 5.3. Modeling diverse data types)
- "The network  $\theta$  takes in the sequence of tokens and outputs a categorical distribution over the v possible values of the next token" (Section 3. Background)
- "we restricted our investigation to a class of sparse attention patterns that have connectivity between all positions over several steps of attention." (Section 4.1. Qualitative assessment of learned attention patterns)
- Inference: Set In/Out Dimension to 3D (x, y, z) because images are indexed by row, column, and channel; set In/Out Dynamics to Fixed because CIFAR-10 images are fixed-length byte sequences (3072 bytes). Marked Attention as Static and State as Direct based on predetermined sparse attention patterns and next-token prediction described above.

### Task: Text density modeling (autoregressive generation)
- "We empirically test our architecture on density modeling tasks including natural images, text, and raw audio." (Section 7. Experiments)
- "we trained models on the EnWik8 dataset, which represents the first  $10^8$  bytes of Wikipedia" (Section 7.2. Text)
- "We trained with a context length of 12,288" (Section 7.2. Text)
- "We treat images, text, and audio as a sequence of discrete tokens, typically raw bytes." (Section 3. Background)
- "The network  $\theta$  takes in the sequence of tokens and outputs a categorical distribution over the v possible values of the next token" (Section 3. Background)
- "we restricted our investigation to a class of sparse attention patterns that have connectivity between all positions over several steps of attention." (Section 4.1. Qualitative assessment of learned attention patterns)
- Inference: Set In/Out Dimension to 1D (t) because text is treated as a token sequence; set In/Out Dynamics to Capped based on the stated context length. Marked Attention as Static and State as Direct based on predetermined attention patterns and next-token prediction described above.

### Task: Raw audio density modeling (autoregressive generation)
- "We empirically test our architecture on density modeling tasks including natural images, text, and raw audio." (Section 7. Experiments)
- "we trained models on the classical music dataset released by (Dieleman et al., 2018)." (Section 7.4. Classical music from raw audio)
- "Samples are available for sequences of length 65,536, which correspond to around 5 seconds of generated audio at 12kHz." (Section 7.4. Classical music from raw audio)
- "We treat images, text, and audio as a sequence of discrete tokens, typically raw bytes." (Section 3. Background)
- "The network  $\theta$  takes in the sequence of tokens and outputs a categorical distribution over the v possible values of the next token" (Section 3. Background)
- "we restricted our investigation to a class of sparse attention patterns that have connectivity between all positions over several steps of attention." (Section 4.1. Qualitative assessment of learned attention patterns)
- Inference: Set In/Out Dimension to 1D (t) because raw audio is treated as a token sequence; set In/Out Dynamics to Capped based on the fixed sequence lengths reported. Marked Attention as Static and State as Direct based on predetermined attention patterns and next-token prediction described above.
