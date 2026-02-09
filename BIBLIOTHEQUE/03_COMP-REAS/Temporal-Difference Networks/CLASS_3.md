# Temporal-Difference Networks (Not specified in the paper)
Source: Temporal-Difference Networks.md

## Core reasons
- The paper explicitly identifies a limitation of conventional TD methods and proposes a new mechanism (TD networks) that changes how predictions are computed via interrelated prediction nodes.
- The contribution is an algorithmic/computational framework (question network + answer network + TD updates) enabling prediction capabilities beyond standard TD, including fixed-interval, action-conditional, and non-Markov predictive-state learning.

## Evidence extracts
- "Rather than relating a single prediction to itself at a later time, as in conventional TD methods, a TD network relates each prediction in a set of predictions to other predictions in the set at a later time." (Abstract)
- "This is one of the simplest cases that cannot otherwise be handled by TD methods." (Section 1 The Learning-to-predict Problem)
- "A *TD network* is a network of nodes, each representing a single scalar prediction." (Section 2 TD Networks)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
