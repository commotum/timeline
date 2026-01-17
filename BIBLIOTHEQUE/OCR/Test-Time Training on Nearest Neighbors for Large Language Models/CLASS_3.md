# TEST-TIME TRAINING ON NEAREST NEIGHBORS FOR LARGE LANGUAGE MODELS (Not specified in the paper.)
Source: Test-Time Training on Nearest Neighbors for Large Language Models.md

## Core reasons
- Proposes a test-time training mechanism that fine-tunes the model per test instance using retrieved neighbors, changing how computation happens at inference.
- Focuses on a procedure for adapting the model at test time rather than introducing new positional encodings, higher-dimensional modeling, or new datasets.

## Evidence extracts
- "To avoid these complications, we simply fine-tune the model on retrieved data at test time, using its standard training setup." (Abstract)
- "For each test instance, we retrieve its nearest neighbors from a huge database, and fine-tune the model on those neighbors before applying it to the test instance." (Section 1 Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
