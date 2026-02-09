# An Emphatic Approach to the Problem of Off-policy Temporal-Difference Learning (Not specified in the paper.)
Source: An Emphatic Approach to the Problem of Off-policy Temporal-Difference Learning.md

## Core reasons
- The paper’s main contribution is a new TD learning method (emphatic $TD(\lambda)$) with a formal stability result for off-policy learning under linear function approximation, which is a methodological/theoretical ML contribution.
- The work is not centered on positional encoding, transformer dimensional adaptation, or dataset/benchmark creation; its focus is reinforcement-learning stability and convergence principles.

## Evidence extracts
- "In particular, we show that varying the emphasis of linear  $TD(\lambda)$ 's updates in a particular way causes its expected update to become stable under off-policy training." (Abstract)
- "If we further assume that  $i(s) > 0, \forall s \in \mathcal{S}$ , then the column sums are all positive, the key matrix is positive definite, and emphatic  $TD(\lambda)$  and its expected update are stable." (Section 6. Off-policy Stability of Emphatic $TD(\lambda)$)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
