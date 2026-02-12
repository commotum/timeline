# Length Generalization of Causal Transformers without Position Encoding (Year not specified)
Source: Length Generalization of Causal Transformers without Position Encoding.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly centers the paper on "Transformer-based language models" and "Transformers without position encodings (NoPE)," indicating Transformers are the primary model family.
- The method directly tunes "attention heads' best temperature hyper-parameters," which is a core self-attention mechanism rather than a peripheral baseline detail.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract and available auxiliary files were sufficient and consistent for a high-confidence decision.

## Evidence
- "Generalizing to longer sentences is important for recent Transformer-based language models. Besides algorithms manipulating explicit position features, the success of Transformers without position encodings (NoPE) provides a new way to overcome the challenge." (Abstract, Length Generalization of Causal Transformers without Position Encoding.md:14)
- "We propose a parameterefficient tuning for searching attention heads' best temperature hyper-parameters, which substantially expands NoPE's context size." (Abstract, Length Generalization of Causal Transformers without Position Encoding.md:14)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Decision was clear from the abstract plus TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided unambiguous Transformer/self-attention evidence.
