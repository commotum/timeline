# How I came in first on ARC-AGI-Pub using Sonnet 3.5 with Evolutionary Test-time Compute (2024)
Source: Evolutionary Test-Time Compute.md

## Core reasons
- Proposes an evolutionary, iterative test-time compute procedure where an LLM generates, evaluates, and refines candidate program functions to solve ARC challenges.
- Frames scaled test-time compute guided by evolutionary principles as the mechanism to overcome LLM reasoning limitations, which is a computation-focused proposal.

## Evidence extracts
- "My approach works by having Sonnet 3.5 generate a bunch of Python transform functions, testing them against challenge examples, and then using the bestperforming functions to create new prompts for generating even better solutions. This process repeats multiple times, ultimately generating up to 500 functions using 31 dynamic prompts per challenge." (Section introduction)
- "LLMs can compensate for their generalization limitations through scaled test-time compute guided by evolutionary principles." (Section introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
