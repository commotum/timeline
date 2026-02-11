# Accelerating Search-Based Program Synthesis using Learned Probabilistic Models (Euphony) (2018)
Source: TASK-DOMAINS.md, TASK_MODEL_RATIO.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: hint-only

## Why
- The hints describe Euphony’s core method as learned probabilistic/statistical program models for grammar-guided synthesis, not a Transformer/self-attention network.
- Reported model structure is domain-specific term/predicate statistical models used with search heuristics and divide-and-conquer enumeration, with no Transformer-style blocks indicated.

## Evidence
- "The paper covers program synthesis tasks across three domains: string manipulation, bit-vector manipulation, and circuit transformation." (TASK-DOMAINS.md, Summary)
- "It takes two statistical program models: the term model  $G_q^T$  and the predicate model  $G_q^P$ , and the two heuristic functions based on those grammars, respectively." (TASK_MODEL_RATIO.md, Section 3.4.2, "Divide-and-Conquer Enumeration")

## Pass accounting
Pass 0 (hint-first): performed - sufficient evidence for high-confidence NO from hint files (statistical/probabilistic synthesis models, no self-attention cues).
Pass 1 (source triage): skipped - hint evidence sufficient.
Pass 2 (source deep dive): skipped - not needed after Pass 0.
