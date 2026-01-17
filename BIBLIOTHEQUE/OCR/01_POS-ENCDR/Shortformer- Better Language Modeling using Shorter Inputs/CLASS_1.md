# Shortformer: Better Language Modeling Using Shorter Inputs (Not specified in the paper.)
Source: Shortformer- Better Language Modeling using Shorter Inputs.md

## Core reasons
- The paper centers on a transformer language model and introduces a positional-information change (position-infused attention) that alters how position embeddings are applied in attention.
- It explicitly critiques existing relative position embeddings as costly and replaces them with a new absolute-position embedding placement to enable caching.

## Evidence extracts
- "Existing methods require computationally expensive relative position embeddings; we introduce a simple alternative of adding absolute position embeddings to queries and keys instead of to word embeddings, which efficiently produces superior results." (Abstract)
- "model so that it does not add position embeddings at the *beginning* of the computation (step 2), but rather adds them to the query and key vectors at each layer (but *not* to the value vectors)." (Section 5.1 Position-Infused Attention (PIA))

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
