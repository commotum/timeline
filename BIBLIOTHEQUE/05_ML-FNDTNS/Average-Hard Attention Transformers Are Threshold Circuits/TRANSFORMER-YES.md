# Average-Hard Attention Transformers are Constant-Depth Uniform Threshold Circuits (Year not specified)
Source: Average-Hard Attention Transformers Are Threshold Circuits.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract identifies the central model family as transformers with average-hard attention, not as a peripheral baseline or related-work mention.
- Auxiliary analyses consistently frame the core result around what "a transformer with average-hard attention" can decide, confirming self-attention-based Transformer architecture is material to the main claim.
- The extending-dimensions analysis markdown was unavailable (provided as `MISSING`), but available Pass 1 sources were sufficient for a high-confidence decision.

## Evidence
- "Transformers have emerged as a widely used neural network model for various natural language processing tasks." (Average-Hard Attention Transformers Are Threshold Circuits.md, Abstract)
- "Merrill et al. (2022) prove that average-hard attention transformers recognize languages that fall within the complexity class TC<sup>0</sup>" (Average-Hard Attention Transformers Are Threshold Circuits.md, Abstract)
- "Every language that can be decided by a transformer with average-hard attention is in uniform  $TC^0$ ." (TASK-DOMAINS.md, Evidence section quoting Theorem 2, Section 3 Main result)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence TRANSFORMER-YES.
Pass 2 (targeted source scan): skipped - Pass 1 evidence was already sufficient.
