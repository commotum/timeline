# Goal-Aware Neural SAT Solver (Not specified in the paper.)
Source: Goal-Aware Neural SAT Solver (QuerySAT - goal-aware guidance).md

## Core reasons
- Proposes a query mechanism that alters computation by iteratively generating solution trials, evaluating them with an unsupervised loss, and updating the network state based on feedback.
- Presents QuerySAT as a step-wise recurrent neural SAT solver that uses the query mechanism to improve reasoning over SAT instances.

## Evidence extracts
- "In this paper, we introduce a step-wise recurrent neural SAT solver that at each step comes up with a query of variable assignments, evaluates it with an unsupervised loss, and updates its state based on the evaluation results." (Section I. INTRODUCTION)
- "The proposed query mechanism works by producing a query, evaluating it using an unsupervised loss function, and passing the resulting value back to the neural network for interpretation." (Fig. 1)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
