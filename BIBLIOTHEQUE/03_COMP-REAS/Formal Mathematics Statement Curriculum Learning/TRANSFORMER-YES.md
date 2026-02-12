# Formal Mathematics Statement Curriculum Learning (Year not specified)
Source: Formal Mathematics Statement Curriculum Learning.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The paper explicitly states that its core model is Transformer-based (decoder-only, GPT-3-style).
- This model is the one used for the main expert-iteration results, so attention is central rather than peripheral.
- The Extending-dimensions analysis markdown was unavailable (`MISSING`), so the decision used the abstract, available auxiliary files, and a targeted source scan.

## Evidence
- "We use decoder-only Transformers similar to GPT-3 (Brown et al., 2020)." (Formal Mathematics Statement Curriculum Learning.md, Section 4.1 "Model")
- "we propose an expert iteration methodology for GPT-f (Polu & Sutskever, 2020)" (Formal Mathematics Statement Curriculum Learning.md, Section 1.2 "Contribution")

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Read the abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; evidence suggested GPT-style language-model theorem proving but lacked an explicit Transformer statement.
Pass 2 (targeted source scan): performed - Scanned the model section and found explicit architecture evidence ("decoder-only Transformers similar to GPT-3").
