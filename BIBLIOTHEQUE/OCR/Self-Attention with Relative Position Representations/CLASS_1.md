# Self-Attention with Relative Position Representations (Not specified in the paper.)
Source: Self-Attention with Relative Position Representations.md

## Core reasons
- The paper targets the Transformer self-attention mechanism and replaces absolute positional encodings with a relative position representation mechanism.
- It critiques reliance on absolute position inputs and proposes a new relative-position encoding inside attention to improve translation performance.

## Evidence extracts
- "Instead, it requires adding representations of absolute positions to its inputs. In this work we present an alternative approach, extending the self-attention mechanism to efficiently consider representations of the relative positions, or distances between sequence elements." (Abstract)
- "In this work we present an efficient way of incorporating relative position representations in the self-attention mechanism of the Transformer. Even when entirely replacing its absolute position encodings, we demonstrate significant improvements in translation quality on two machine translation tasks." (Section 1 Introduction)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
