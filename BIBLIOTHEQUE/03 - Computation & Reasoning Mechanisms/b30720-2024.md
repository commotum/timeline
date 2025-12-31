# Solving olympiad geometry without human demonstrations (2024)
Source: b30720-2024.pdf

## Core reasons
- The paper proposes AlphaGeometry, a theorem prover that couples a language model with a symbolic deduction engine in an iterative proof-search loop, which is a computation and reasoning mechanism contribution.
- It focuses on generating auxiliary constructions during search when symbolic deduction stalls, changing how computation proceeds rather than presenting a dataset or positional encoding.

## Evidence extracts
- "b, AlphaGeometry initiates the proof search by running the symbolic deduction engine. The engine exhaustively deduces new statements from the theorem premises until the theorem is proven or new statements are exhausted. c, Because the symbolic engine fails to find a proof, the language model constructs one auxiliary point, growing the proof state before the symbolic engine retries. The loop continues until a solution is found." (p. 2)
- "On a high level, proof search is a loop in which the language model and the symbolic deduction engine take turns to run" (p. 3)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
