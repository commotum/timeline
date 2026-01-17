# ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision (2021)
Source: ViLT- Vision-and-Language Transformer Without Convolution or Region Supervision.md

## Core reasons
- Proposes a transformer-based vision-and-language model that processes image inputs in a transformer-friendly sequence form rather than via CNNs, enabling 2D visual data to be handled alongside text.
- Uses patch projection of image regions into token sequences so the transformer can model higher-dimensional (image) inputs directly, which is central to the contribution.

## Evidence extracts
- "In this paper, we present a minimal VLP model, Vision-and-Language Transformer (ViLT), monolithic in the sense that the processing of visual inputs is drastically simplified to just the same convolution-free manner that we process textual inputs." (Abstract)
- "**Patch Projection.** To minimize overhead, we adopt the simplest visual embedding scheme: *linear projection* that operates on image patches." (Section 2.3. Visual Embedding Schema)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
