# Beyond $A^*$ : Better Planning with Transformers via Search Dynamics Bootstrapping (Not specified in the paper.)
Source: Beyond A-- Planning with Transformers.md

## Core reasons
- Proposes training Transformers to imitate and improve $A^*$ search dynamics for planning, introducing a new computation mechanism rather than a positional encoding or dimensional adaptation.
- Introduces Searchformer with search dynamics bootstrapping to generate shorter search traces while still producing optimal plans.

## Evidence extracts
- "We demonstrate how to train Transformers to robustly solve complex planning tasks. Similar to LLMs, we train Transformers to predict the next word given a sequence of words. Our experiments use synthetically generated datasets with a synthetic language and vocabulary. Using this framework, we demonstrate how to construct training data such that the resulting model imitates the computation performed by  $A^*$  search (Russell & Norvig, 2021, Chapter 3)." (Section "Our work")
- "Once a model is trained to imitate the search dynamics of non-deterministic  $A^*$  search, it is used to generate a new training dataset consisting of shorter token sequences." (Section "3.3 Moving past algorithm imitation via search dynamics bootstrapping")

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
