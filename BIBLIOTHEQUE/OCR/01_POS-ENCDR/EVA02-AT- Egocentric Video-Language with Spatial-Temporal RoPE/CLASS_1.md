# EVA02-AT: Egocentric Video-Language Understanding with Spatial-Temporal Rotary Positional Embeddings and Symmetric Optimization (Not specified in the paper.)
Source: EVA02-AT- Egocentric Video-Language with Spatial-Temporal RoPE.md

## Core reasons
- The paper critiques existing 3D RoPE positional encoding for spatial-temporal modeling due to manual dimension splitting.
- The core contribution introduces spatial-temporal RoPE that applies rotary positional embeddings across the entire hidden dimension with joint attention.

## Evidence extracts
- "Ineffective spatial-temporal encoding due to manually split 3D rotary positional embeddings that hinder feature interactions" (Abstract)
- "In this way, we thus apply the spatial RoPE and temporal RoPE on the entire dimension instead of manually dividing the dimension into uneven slides." (Section IV.A. EVA-02 AT Transformer)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
