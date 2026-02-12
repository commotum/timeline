# Efficient Evolutionary Program Synthesis (2025)
Source: Efficient Evolutionary Program Synthesis.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The main method is explicitly an "LLM-assisted program synthesis system," and its architecture repeatedly prompts an LLM to generate candidate programs, making the LLM component central to the reported results.
- The auxiliary model-ratio analysis also identifies a single central model instance as this LLM-assisted system across tasks, reinforcing that the core solver is an LLM-centered hybrid rather than a Transformer-free baseline.

## Evidence
- "I designed a <u>DreamCoder</u>-inspired, LLM-assisted program synthesis system that can solve increasingly harder tasks by leveraging learned concepts in an expanding library of programs." (Opening text, Efficient Evolutionary Program Synthesis.md:11)
- "Starting from an empty library, my system loops through each task to prompt an LLM for Python program(s) that can solve all of the training examples." (Architecture section, Efficient Evolutionary Program Synthesis.md:125)
- "I designed a <u>DreamCoder</u>-inspired, LLM-assisted program synthesis system that can solve increasingly harder tasks by leveraging learned concepts in an expanding library of programs." (Quoted evidence, TASK_MODEL_RATIO.md:11)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Decision was supported by the abstract/opening and full reads of TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; Extending-dimensions analysis markdown was unavailable (MISSING).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient to finalize.
