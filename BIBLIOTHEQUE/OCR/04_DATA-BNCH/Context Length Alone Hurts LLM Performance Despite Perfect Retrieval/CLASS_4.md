# Context Length Alone Hurts LLM Performance Despite Perfect Retrieval (Not specified in the paper.)
Source: Context Length Alone Hurts LLM Performance Despite Perfect Retrieval.md

## Core reasons
- The paper centers on controlled evaluations showing that long-context performance degrades even under perfect retrieval, framing a measurement critique rather than a new model or encoding.
- It constructs a synthetic long-context benchmark by extending short-context tasks to study context-length effects across math, QA, and coding.

## Evidence extracts
- "Our systematic experiments across 5 open- and closedsource LLMs on math, question answering, and coding tasks reveal that, even when models can perfectly retrieve all relevant information, their performance still degrades substantially (13.9%–85%) as input length increases but remains well within the models' claimed lengths." (Abstract)
- "Given a pair of evidence and question, we insert distraction tokens in between to reach desired context lengths. This creates input of the form: [Evidence] [Distraction Tokens] [Question]." (Section 3.1 A Long-Context Synthetic Benchmark Covering Math, QA, and Coding)

## Classification
Class name: Data, Benchmarks & Measurement
Class code: 4

$$
\boxed{4}
$$
