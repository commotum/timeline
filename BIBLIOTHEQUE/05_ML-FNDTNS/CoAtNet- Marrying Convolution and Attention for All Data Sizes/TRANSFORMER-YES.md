# CoAtNet: Marrying Convolution and Attention for All Data Sizes (Year not specified)
Source: CoAtNet- Marrying Convolution and Attention for All Data Sizes.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract states CoAtNet is a hybrid architecture that unifies depthwise convolution with self-attention and stacks convolution and attention layers as the core method.
- Auxiliary analysis files identify self-attention as part of the central architecture; the Extending-dimensions analysis markdown was unavailable (`MISSING`) but not needed for confidence.

## Evidence
- "we present CoAtNets (pronounced \"coat\" nets), a family of hybrid models built from two key insights: (1) depthwise Convolution and self-Attention can be naturally unified via simple relative attention; (2) vertically stacking convolution layers and attention layers in a principled way is surprisingly effective in improving generalization, capacity and efficiency." (Abstract, `CoAtNet- Marrying Convolution and Attention for All Data Sizes.md`)
- "self-attention allows the receptive field to be the entire spatial locations and computes the weights based on the re-normalized pairwise similarity between the pair  $(x_i, x_j)$" (Section 2.1 quote recorded in `TASK-DOMAINS.md`, Evidence)
- "Our experiments focus on image classification." (Section 4.1 quote recorded in `TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence YES decision from the abstract plus `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; Extending-dimensions analysis markdown was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient to finalize.
