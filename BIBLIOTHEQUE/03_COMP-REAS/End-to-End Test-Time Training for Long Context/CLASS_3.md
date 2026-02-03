# End-to-End Test-Time Training for Long Context (Not specified in the paper.)
Source: End-to-End Test-Time Training for Long Context.md

## Core reasons
- The paper reframes long-context modeling as continual learning and proposes test-time training that updates weights during inference, changing how computation happens.
- It introduces an end-to-end meta-learning setup to optimize the model for test-time learning, emphasizing a new computation/training mechanism rather than positional encoding or new data.

## Evidence extracts
- "We formulate long-context language modeling as a problem in continual learning rather than architecture design." (Section 1 Introduction)
- "our model continues learning at test time via next-token prediction on the given context, compressing the context it reads into its weights." (Abstract)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
