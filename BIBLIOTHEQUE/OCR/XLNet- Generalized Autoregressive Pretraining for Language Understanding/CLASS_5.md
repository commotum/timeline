# XLNet: Generalized Autoregressive Pretraining for Language Understanding (Not specified in the paper.)
Source: XLNet- Generalized Autoregressive Pretraining for Language Understanding.md

## Core reasons
- The paper's main contribution is a new autoregressive pretraining objective (permutation language modeling) to capture bidirectional context, which is a training-method contribution rather than a dataset or positional encoding change.
- It introduces architectural changes (two-stream attention, Transformer-XL integration) to make the objective work, aligning with ML modeling principles and training methodology.

## Evidence extracts
- "we propose XLNet, a generalized autoregressive pretraining method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order and (2) overcomes the limitations of BERT thanks to its autoregressive formulation." (Abstract)
- "To resolve such a contradiction, we propose to use two sets of hidden representations instead of one:" (Section 2.3)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
