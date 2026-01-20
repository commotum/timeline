# Recursive Language Models (Not specified in the paper.)
Source: Recursive Language Models.md

## Core reasons
- Proposes a new inference-time mechanism where the model treats the prompt as an external environment and recursively calls itself to handle long contexts.
- Focuses on changing how computation happens (REPL interaction and recursive sub-calls) rather than positional encoding, dimensional lifting, or datasets.

## Evidence extracts
- "We propose **Recursive Language Models** (**RLMs**), a general inference strategy that treats long prompts as part of an external *environment* and allows the LLM to *programmatically* examine, decompose, and recursively call itself over snippets of the prompt." (Section **ABSTRACT**)
- "The key insight is that long prompts should not be fed into the neural network (e.g., Transformer) directly but should instead be treated as *part of the environment that the LLM can symbolically interact with*." (Section 1 Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
