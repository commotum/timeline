# Zero-Shot Text-to-Image Generation (Not specified in the paper.)
Source: Zero-Shot Text-to-Image Generation.md

## Core reasons
- The paper's method centers on a transformer that models text and image tokens in a single autoregressive stream.
- It adapts images into a 32 x 32 token grid and concatenates those tokens with text to enable transformer-based modeling of image data.

## Evidence extracts
- "We describe a simple approach for this task based on a transformer that autoregressively models the text and image tokens as a single stream of data." (Abstract)
- "We concatenate up to 256 BPE-encoded text tokens with the  $32 \times 32 = 1024$  image tokens, and train an autoregressive transformer to model the joint distribution over the text and image tokens." (Section 2. Method)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
