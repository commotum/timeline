# VideoBERT: A Joint Model for Video and Language Representation Learning (Not specified in the paper.)
Source: VideoBERT- A Joint Model for Video and Language Representation Learning.md

## Core reasons
- The paper adapts BERT to jointly model video and language by representing video as discrete tokens and applying a transformer to those sequences.
- The contribution centers on extending transformer modeling to video (a higher-dimensional domain) rather than proposing new positional encodings or datasets.

## Evidence extracts
- "we build upon the BERT model to learn bidirectional joint distributions over sequences of visual and linguistic tokens, derived from vector quantization of video data and off-the-shelf speech recognition outputs, respectively." (Abstract)
- "To extend BERT to video, in such a way that we may still leverage pretrained language models and scalable implementations for inference and learning, we decided to make minimal changes, and transform the raw visual data into a discrete sequence of tokens." (Section 3.2. The VideoBERT model)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
