# Flamingo: a Visual Language Model for Few-Shot Learning (Not specified in the paper)
Source: Flamingo- a Visual Language Model for Few-Shot Learning.md

## Core reasons
- The paper's central contribution is a visual language model architecture that connects pretrained vision and language components to handle visual inputs alongside text, which is a transformer adaptation to higher-dimensional (image/video) data.
- It emphasizes processing sequences of interleaved images/videos and text for multimodal tasks, indicating the primary innovation is enabling transformers to model visual domains rather than altering positional encoding or proposing new computation mechanisms.

## Evidence extracts
- "We introduce Flamingo, a family of Visual Language Models (VLM) with this ability. We propose key architectural innovations to: (i) bridge powerful pretrained vision-only and language-only models, (ii) handle sequences of arbitrarily interleaved visual and textual data, and (iii) seamlessly ingest images or videos as inputs." (Abstract)
- "This section describes Flamingo: a visual language model that accepts text interleaved with images/videos as input and outputs free-form text." (Section 2 Approach)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
