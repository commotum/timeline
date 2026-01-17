# Spherical Position Encoding for Transformers (Not specified in the paper.)
Source: Spherical Position Encoding for Transformers.md

## Core reasons
- The paper proposes a new positional encoding mechanism by extending RoPE to spherical coordinates for geotokens, directly modifying how positions are encoded in Transformers.
- It highlights that sequential position is not appropriate for geotokens and frames traditional positional embeddings as limited for spatial/geographical settings.

## Evidence extracts
- "In order to induce the concept of relative position for such a setting and maintain the proportion between the physical distance and distance on embedding space, we formulate a position encoding mechanism based on RoPE architecture which is adjusted for spherical coordinates." (Abstract)
- "Recognizing the limitations of traditional position embeddings in this spatial context, we employed and extended the Rotary Position Embedding (RoPE) mechanism to accommodate spherical coordinates, aligning the transformer architecture to work seamlessly with geographical data." (Conclusion)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
