# MaskGIT: Masked Generative Image Transformer (Not specified in the paper.)
Source: MaskGIT- Masked Generative Image Transformer.md

## Core reasons
- Proposes a bidirectional transformer with masked visual token modeling and an iterative, parallel decoding procedure, changing how generation is computed.
- The main contribution is a new non-autoregressive decoding mechanism for image synthesis rather than positional encoding changes or dataset creation.

## Evidence extracts
- "This paper proposes a novel image synthesis paradigm using a bidirectional transformer decoder, which we term MaskGIT. During training, MaskGIT learns to predict randomly masked tokens by attending to tokens in all directions. At inference time, the model begins with generating all tokens of an image simultaneously, and then refines the image iteratively conditioned on the previous generation." (Abstract)
- "We introduce a novel decoding method where all tokens in the image are generated simultaneously in parallel." (Section 3.2. Iterative Decoding)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
