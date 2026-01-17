# VISUALBERT: A SIMPLE AND PERFORMANT BASELINE FOR VISION AND LANGUAGE (Not specified in the paper.)
Source: VisualBERT- A Simple and Performant Baseline for Vision and Language.md

## Core reasons
- The paper adapts a Transformer (BERT) to jointly process text with image regions, enabling a vision-language model rather than a text-only Transformer.
- It treats image regions as input tokens/visual embeddings and feeds them into the Transformer with text, which is a higher-dimensional (image) domain adaptation.

## Evidence extracts
- "VisualBERT consists of a stack of Transformer layers that implicitly align elements of an input text and regions in an associated input image with self-attention." (Abstract)
- "image features extracted from object proposals are treated as unordered input tokens and fed into VisualBERT along with text." (Section 1 Introduction)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
