# Training Verifiers to Solve Math Word Problems (Not specified in the paper.)
Source: GSM8K- Training Verifiers to Solve Math Word Problems.md

## Core reasons
- The paper identifies a missing capability in autoregressive reasoning (no self-correction) and positions verification as the remedy.
- The core contribution is a verifier-based inference mechanism that ranks many candidate solutions and selects the highest-scoring one.

## Evidence extracts
- "When generating a solution, autoregressive models have no mechanism to correct their own errors. Solutions that veer off-course quickly become unrecoverable." (Section 1 Introduction)
- "We propose training verifiers to evaluate the correctness of model generated solutions, similar to concurrent work by Shen et al. (2021a). At test time, we sample a fixed number of candidate solutions and select the solution ranked highest by the verifier." (Section 1 Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
