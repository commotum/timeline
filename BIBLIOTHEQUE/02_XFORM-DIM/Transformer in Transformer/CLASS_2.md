# Transformer in Transformer (Not specified in the paper.)
Source: Transformer in Transformer.md

## Core reasons
- The paper adapts Transformer processing for 2D images by structuring images into patches and sub-patches (visual sentences/words) to model visual data.
- The main contribution is a vision-specific Transformer architecture (TNT) that captures global and local image information, indicating a Transformer adaptation for higher-dimensional visual domains.

## Evidence extracts
- "we regard the local patches (e.g.,  $16 \times 16$ ) as \"visual sentences\" and present to further divide them into smaller patches (e.g.,  $4\times4$ ) as \"visual words\"." (Abstract)
- "Given a 2D image, we uniformly split it into n patches... Instead, we propose Transformer-iN-Transformer (TNT) architecture to learn both global and local information in an image." (Section 2.2 Transformer in Transformer)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
