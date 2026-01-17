# SELF-CONSISTENCY IMPROVES CHAIN OF THOUGHT REASONING IN LANGUAGE MODELS (Not specified in the paper.)
Source: Self-Consistency Improves Chain of Thought Reasoning in Language Models.md

## Core reasons
- Proposes a new decoding strategy that changes inference computation for chain-of-thought reasoning rather than altering positional encoding or data.
- The method samples multiple reasoning paths and marginalizes over them to select the most consistent answer, targeting reasoning performance improvements.

## Evidence extracts
- "In this paper, we propose a new decoding strategy, *self-consistency*, to replace the naive greedy decoding used in chain-of-thought prompting." (Abstract)
- "we propose a \"sample-and-marginalize\" decoding procedure: we first *sample* from the language model's decoder to generate a *diverse* set of reasoning paths; each reasoning path might lead to a different final answer, so we determine the optimal answer by *marginalizing out* the sampled reasoning paths to find the most consistent answer in the final answer set." (Section 1 Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
