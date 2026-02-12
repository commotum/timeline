# The Rotary Position Embedding May Cause Dimension Inefficiency in Attention Heads for Long-Distance Retrieval (Year not specified)
Source: The Rotary Position Embedding May Cause Dimension Inefficiency.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract centers on RoPE behavior in attention heads of large language models, which are Transformer-style self-attention architectures.
- Auxiliary analyses describe the evaluated tasks as long-distance retrieval and long-context QA using LLM attention heads, reinforcing that self-attention is core to the main results.
- Extending-dimensions analysis markdown was unavailable (`MISSING`), but the available abstract and TASK-* files were sufficient for a high-confidence decision.

## Evidence
- "The Rotary Position Embedding (RoPE) is widely used in the attention heads of many large language models (LLM)." (Abstract, The Rotary Position Embedding May Cause Dimension Inefficiency.md)
- "the attention head can retrieve  $v_i$  with  $q_i$" (Task evidence, Section 4 Controlled Experiment, TASK-DOMAINS.md)
- "As we hypothesize that the dimension inefficiency only occurs for attention heads that model long dependency, we choose a task that involves long dependence modeling, the long-context question-answering task." (§5.1 quote in TASK_MODEL_RATIO.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence TRANSFORMER-YES decision.
Pass 2 (targeted source scan): skipped - Pass 1 evidence was already sufficient.
