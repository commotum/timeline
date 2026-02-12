# Convolutional Sequence to Sequence Learning (Year not specified)
Source: Convolutional Sequence to Sequence Learning.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract states the proposed model is "based entirely on convolutional neural networks," which is a direct non-Transformer architectural cue.
- The paper describes attention in decoder layers, but the core architecture is ConvS2S (stacked convolutions with kernels), not Transformer-style self-attention blocks.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "We introduce an architecture based entirely on convolutional neural networks." (Abstract, Convolutional Sequence to Sequence Learning.md)
- "In this paper we propose an architecture for sequence to sequence modeling that is entirely convolutional." (Section 1 Introduction, Convolutional Sequence to Sequence Learning.md)
- "This instance of our architecture has 20 layes in the encoder and 20 layers in the decoder, both using kernels of width 3 and hidden size 512 throughout." (Section 5.1 quote in TASK_MODEL_RATIO.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence TRANSFORMER-NO from abstract and auxiliary model-family cues.
Pass 2 (targeted source scan): skipped - not needed because Pass 1 was already decisive.
