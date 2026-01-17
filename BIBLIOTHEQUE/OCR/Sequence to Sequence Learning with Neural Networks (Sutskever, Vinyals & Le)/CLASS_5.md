# Sequence to Sequence Learning with Neural Networks (Not specified in the paper.)
Source: Sequence to Sequence Learning with Neural Networks (Sutskever, Vinyals & Le).md

## Core reasons
- Proposes an end-to-end sequence-to-sequence LSTM architecture for mapping input sequences to output sequences, which is a core modeling contribution rather than positional encoding or dataset work.
- Focuses on architectural and training choices for deep LSTMs in machine translation, fitting a general ML modeling contribution outside classes 1-4.

## Evidence extracts
- "In this paper, we present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. Our method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector." (Abstract)
- "The goal of the LSTM is to estimate the conditional probability  $p(y_1,\ldots,y_{T'}|x_1,\ldots,x_T)$  where  $(x_1,\ldots,x_T)$  is an input sequence and  $y_1,\ldots,y_{T'}$  is its corresponding output sequence whose length T' may differ from T." (Section 2 The model)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
