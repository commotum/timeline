# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020)
Source: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract defines RAG around a pre-trained seq2seq parametric model plus neural retriever; the auxiliary analysis ties this setup to BART/T5-style seq2seq backbones, which are Transformer architectures using self-attention.
- The auxiliary files indicate this same RAG setup is used across the paper’s evaluated tasks, so Transformer-based components are central to the main results; the extending-dimensions file was unavailable.

## Evidence
- "We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever." (Abstract, `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.md`)
- "Like T5 [51] or BART, RAG can be fine-tuned on any seq2seq task, whereby both the generator and retriever are jointly learned." (`TASK_MODEL_RATIO.md`, verbatim evidence citing Section 1 Introduction)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence decision; extending-dimensions analysis file was unavailable.
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient.
