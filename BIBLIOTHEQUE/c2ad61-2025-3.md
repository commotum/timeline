# From Parrots to Von Neumanns: How Evolutionary Test-Time Compute Achieved State-of-the-Art on ARC-AGI (2025)
Source: c2ad61-2025.pdf

## Core reasons
- Introduces Evolutionary Test-Time Compute (ETTC), a test-time evolutionary search mechanism that treats LLMs as hypothesis generators and selects/refines candidates via execution-based fitness.
- Frames ARC-AGI performance limits as a reasoning/generalization gap and addresses it through explicit test-time compute (generation, verification, revision) rather than new data or positional encoding.

## Evidence extracts
- "I introduce Evolutionary Test-Time Compute (ETTC), a framework that treats language models as hypothesis generators and uses training examples as fitness functions. The approach: generate diverse solution candidates, score them by execution, select the best performers, and evolve better solutions through explicit revision with error feedback." (p. 1)
- "The core insight was treating test-time compute as evolution: generate diverse hypotheses, verify them against training examples, and breed better solutions from the fittest candidates." (p. 4)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
