# Neural Machine Translation by Jointly Learning to Align and Translate (Not specified in the paper.)
Source: Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau, Cho & Bengio).md

## Core reasons
- Introduces a soft alignment/attention mechanism that (soft-)searches source positions for each target word, changing how computation happens during decoding.
- Addresses the fixed-length context vector bottleneck by computing context vectors from a sequence of annotations per decoding step instead of a single vector.

## Evidence extracts
- "In this paper, we conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder—decoder architecture, and propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly." (Abstract)
- "We extended the basic encoder–decoder by letting a model (soft-)search for a set of input words, or their annotations computed by an encoder, when generating each target word. This frees the model from having to encode a whole source sentence into a fixed-length vector, and also lets the model focus only on information relevant to the generation of the next target word." (7 CONCLUSION)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
