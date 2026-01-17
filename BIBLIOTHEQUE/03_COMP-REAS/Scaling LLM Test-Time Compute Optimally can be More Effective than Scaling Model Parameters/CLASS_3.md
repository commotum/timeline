# Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (Not specified in the paper.)
Source: Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.md

## Core reasons
- The paper proposes mechanisms for allocating and scaling inference-time computation, including search against verifiers and adaptive proposal distribution updates.
- It formalizes a compute-optimal test-time strategy that changes how computation is performed per prompt, rather than changing positional encoding or data resources.

## Evidence extracts
- "In this work, we analyze two primary mechanisms to scale test-time computation: (1) searching against dense, process-based verifier reward models; and (2) updating the model's distribution over a response adaptively, given the prompt at test time." (Section 1. Introduction)
- "we define the \"test-time compute-optimal scaling strategy\" as the strategy that chooses hyperparameters corresponding to a given test-time strategy for maximal performance benefits on a given prompt at test time." (Section 3.1. Test-Time Compute-Optimal Scaling Strategy)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
