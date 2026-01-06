# MUSIC TRANSFORMER: GENERATING MUSIC WITH LONG-TERM STRUCTURE (Not specified in the paper.)
Source: Music Transformer.md

## Core reasons
- The paper critiques existing relative positional attention for quadratic memory cost and targets that limitation directly.
- The core contribution is a modified relative positional attention mechanism with a memory-efficient algorithm, improving how positions are handled.

## Evidence extracts
- "Existing approaches for representing relative positional information in the Transformer modulate attention based on pairwise distance (Shaw et al., 2018). This is impractical for long sequences such as musical compositions since their memory complexity for intermediate relative information is quadratic in the sequence length. We propose an algorithm that reduces their intermediate memory requirement to linear in the sequence length." (Abstract)
- "We improve the implementation of relative attention by reducing its intermediate memory requirement from  $O(L^2D)$  to O(LD), with example lengths shown in Table 1." (Section 3.4 Memory efficient implementation of relative position-based attention)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
