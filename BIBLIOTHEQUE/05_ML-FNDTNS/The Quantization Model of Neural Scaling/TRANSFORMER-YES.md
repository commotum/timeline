# The Quantization Model of Neural Scaling (Year not specified)
Source: The Quantization Model of Neural Scaling.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s core empirical analysis on real language models is explicitly based on "decoder-only transformers," making Transformer self-attention a material part of the main results.
- Although the toy proof-of-concept uses a ReLU MLP, Transformer-based LLM experiments are central (not peripheral) in the paper’s evaluation and decomposition analysis.
- The extending-dimensions analysis file was unavailable (`MISSING`) and was skipped.

## Evidence
- "We validate this prediction on toy datasets, then study how scaling curves decompose for large language models." (Abstract, `The Quantization Model of Neural Scaling.md`)
- "(Section 4) \"For our experiments, we use the Pythia model suite from Eleuther AI [29], a set of decoder-only transformers of varying size trained on approximately 300 billion tokens of The Pile [30].\"" (`TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence from abstract + auxiliary files to determine Transformer usage is central in major experiments.
Pass 2 (targeted source scan): skipped - Not needed after high-sufficiency Pass 1 evidence.
