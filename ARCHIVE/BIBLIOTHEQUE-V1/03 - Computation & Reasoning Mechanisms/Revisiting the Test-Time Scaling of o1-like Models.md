# Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities? (2025)
Source: cb40cf-2025.pdf

## Core reasons
- Focuses on test-time scaling behavior in o1-like LLMs and shows longer CoT reasoning does not reliably improve accuracy, indicating a study of inference-time compute.
- Proposes Shortest Majority Vote, a test-time scaling method that changes how inference computation is performed via parallel scaling and length-aware voting.

## Evidence extracts
- "longer CoTs of these o1-like models do not
  consistently enhance accuracy; in fact, correct
  solutions are often shorter than incorrect ones
  for the same questions." (p. 1)
- "Building on these findings, we propose a novel
  test-time scaling method, Shortest Majority Vote,
  which incorporate parallel scaling approaches with
  our insight on sequential scaling." (p. 2)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
