# Positional Encoding Field (Not specified in the paper.)
Source: Positional Encoding Field.md

## Core reasons
- The paper identifies limitations of existing 2D positional encodings for enforcing spatial coherence and motivates changing how position is handled in DiTs.
- The main contribution is a new positional encoding design (PE-Field) that extends and modifies RoPE with depth-aware and hierarchical encodings.

## Evidence extracts
- "This suggests that spatial coherence in DiTs is primarily enforced by positional encodings rather than by explicit token-to-token dependencies" (Section 1. Introduction)
- "we introduce the Positional Encoding Field (PE-Field), which extends positional encodings from the 2D plane to a structured 3D field. PE-Field incorporates depth-aware encodings for volumetric reasoning and hierarchical encodings for fine-grained sub-patch control" (Abstract)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
