# Precise Zero-Shot Dense Retrieval without Relevance Labels (Year not specified)
Source: Precise Zero-Shot Dense Retrieval without Relevance Labels.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract defines HyDE around an instruction-following LM (InstructGPT) plus a Contriever encoder, which are Transformer-family backbones, so self-attention is central to the method.
- `TASK_MODEL_RATIO.md` confirms Contriever/mContriever are the deployed retrieval backbones for the reported tasks; the extending-dimensions file was unavailable (`MISSING`).

## Evidence
- "Given a query, HyDE first zero-shot instructs an instruction-following language model (e.g. InstructGPT) to gen-The docuerate a hypothetical document." (Precise Zero-Shot Dense Retrieval without Relevance Labels.md, Abstract)
- "We use the English-only Contriever model for English retrieval tasks and multilingual mContriever for non-English tasks." (TASK_MODEL_RATIO.md, item 2, verbatim quote)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for high-confidence TRANSFORMER-YES from abstract + TASK files; extending-dimensions analysis was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient.
