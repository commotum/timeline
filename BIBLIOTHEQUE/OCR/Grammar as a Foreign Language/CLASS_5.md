# Grammar as a Foreign Language (Not specified in the paper.)
Source: Grammar as a Foreign Language.md

## Core reasons
- The paper's main contribution is applying a neural sequence-to-sequence LSTM with attention to syntactic constituency parsing and showing strong results, which is an ML modeling contribution rather than a positional encoding or dimensionality change.
- It focuses on model behavior and training efficacy (including data efficiency with attention) rather than introducing a new benchmark or dataset as the primary contribution.

## Evidence extracts
- "In this paper we show that the domain agnostic attention-enhanced sequence-to-sequence model achieves state-of-the-art results on the most widely used syntactic constituency parsing dataset, when trained on a large synthetic corpus that was annotated using existing parsers." (Abstract)
- "We trained a sequence-to-sequence model with attention on the small human-annotated parsing dataset and were able to achieve an F1 score of 88.3 on section 23 of the WSJ without the use of an ensemble and 90.5 with an ensemble, which matches the performance of the BerkeleyParser (90.4) when trained on the same data." (Introduction)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
