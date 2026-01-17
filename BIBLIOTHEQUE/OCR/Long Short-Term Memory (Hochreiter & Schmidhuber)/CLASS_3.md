# Long Short-Term Memory (1997)
Source: Long Short-Term Memory (Hochreiter & Schmidhuber).md

## Core reasons
- Proposes LSTM to fix long-term credit assignment failures caused by vanishing error signals, enabling learning across very long time lags.
- Introduces a new computation mechanism using memory cells with constant error flow and multiplicative input/output gates to control storage and retrieval.

## Evidence extracts
- "Learning to store information over extended time intervals via recurrent backpropagation takes a very long time, mostly due to insufficient, decaying error back flow. We briefly review Hochreiter's 1991 analysis of this problem, then address it by introducing a novel, efficient, gradient-based method called \"Long Short-Term Memory\" (LSTM)." (Abstract)
- "A multiplicative  $input\ gate\ unit$  is introduced to protect the memory contents stored in j from perturbation by irrelevant inputs. Likewise, a multiplicative  $output\ gate\ unit$  is introduced which protects other units from perturbation by currently irrelevant memory contents stored in j." (Section 4 Long Short-Term Memory)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
