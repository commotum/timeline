# AST: Audio Spectrogram Transformer (Not specified in the paper.)
Source: AST- Audio Spectrogram Transformer.md

## Core reasons
- Proposes a Transformer-based architecture for audio classification by operating directly on 2D audio spectrograms, rather than 1D text sequences.
- Central adaptation is converting a 2D spectrogram into a sequence of patches for Transformer processing, enabling modeling in a higher-dimensional domain.

## Evidence extracts
- "In this paper, we answer the question by introducing the Audio Spectrogram Transformer (AST), the first convolution-free, purely attention-based model for audio classification." (Abstract)
- "We then split the spectrogram into a sequence of N 16  $\times$  16 patches with an overlap of 6 in both time and frequency dimension, where  $N = 12 \lceil (100t - 16)/10 \rceil$  is the number of patches and the effective input sequence length for the Transformer." (Section 2.1. Model Architecture)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
