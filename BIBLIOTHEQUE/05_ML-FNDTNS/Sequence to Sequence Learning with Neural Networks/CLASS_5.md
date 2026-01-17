# Sequence to Sequence Learning with Neural Networks (Not specified in the paper.)
Source: Sequence to Sequence Learning with Neural Networks.md

## Core reasons
- Proposes a general end-to-end sequence-to-sequence architecture using LSTMs to encode a variable-length input into a fixed vector and decode the target sequence, which is a modeling/architecture contribution.
- Emphasizes a training/encoding choice (reversing source sentences) to improve optimization and performance, rather than introducing new positional encodings or datasets.

## Evidence extracts
- "Our method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector." (Abstract)
- "Third, we found it extremely valuable to reverse the order of the words of the input sentence." (Section 2 The model)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
