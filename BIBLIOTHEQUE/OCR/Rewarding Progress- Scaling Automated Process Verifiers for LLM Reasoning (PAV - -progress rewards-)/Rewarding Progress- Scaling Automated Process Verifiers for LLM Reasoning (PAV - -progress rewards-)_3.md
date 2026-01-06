# Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning (Not specified in the paper.)
Source: Rewarding Progress- Scaling Automated Process Verifiers for LLM Reasoning (PAV - -progress rewards-).md

## Core reasons
- Proposes process rewards that explicitly measure step-level progress to guide test-time search and online RL, changing how reasoning is computed.
- Introduces process advantage verifiers (PAVs) as a mechanism to provide dense step-level feedback that improves search/RL efficiency.

## Evidence extracts
- "Our key insight is that per-step, process rewards that measure a notion of progress: change in the likelihood of arriving at a correct final answer before and after taking the step, are effective, for both test-time beam search and online RL." (Introduction)
- "To predict the advantages of such provers we train dense verifiers, called *process advantage verifiers* (*PAVs*), that accelerate sample and compute efficiency of RL and search." (Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
