# Test-Time Learning for Large Language Models (2025)
Source: cf25a5-2025.pdf

## Core reasons
- Proposes a test-time learning method that dynamically adapts LLMs using only unlabeled test data, changing inference-time computation to handle distribution shifts.
- Frames the test-time update mechanism as minimizing test/input perplexity to drive self-supervised adaptation, rather than positional encoding or dimensional lifting.

## Evidence extracts
- "Test-Time Learning (TTL) paradigm for LLMs,
namely TLM,whichdynamicallyadaptsLLMs
to target domains using only unlabeled test data
during testing." (p. 1)
- "we propose minimizing the perplexity of test samples
as the optimization objective, which effectively enhances
the performance of LLMs in dynamic environments." (p. 3)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
