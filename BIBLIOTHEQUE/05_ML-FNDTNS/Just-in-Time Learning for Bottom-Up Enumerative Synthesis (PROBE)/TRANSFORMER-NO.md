# Just-in-Time Learning for Bottom-Up Enumerative Synthesis (2020)
Source: Just-in-Time Learning for Bottom-Up Enumerative Synthesis (PROBE).md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract and auxiliary analyses describe PCFG-guided bottom-up enumerative program synthesis, not a Transformer/self-attention architecture.
- No Transformer-style self-attention model is presented as central; the auxiliary task/domain analysis does not identify attention dynamics.
- The extending-dimensions analysis markdown was unavailable (`MISSING`), but the available abstract and auxiliary files are sufficient for a high-confidence decision.

## Evidence
- "we show how to bootstrap one *just in time*, during synthesis, by learning from partial solutions encountered along the way." (Abstract, Just-in-Time Learning for Bottom-Up Enumerative Synthesis (PROBE).md:9)
- "we propose *just-in-time learning*, a novel technique that learns a *probabilistic context-free grammar* (PCFG) for a given synthesis problem \"just in time\", *i.e.* during synthesis, rather than ahead of time." (Introduction, Just-in-Time Learning for Bottom-Up Enumerative Synthesis (PROBE).md:63)
- "program synthesis (string manipulation),context-free grammar (DSL) and input-output examples (strings),Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Constructed,program from the DSL that satisfies the examples,Not specified in the paper.,Not specified in the paper." (TASK-DOMAINS.csv:2)
- "It starts by initializing the PCFG with CFG  $\\mathcal{G}$  and a uniform distribution  $p_u$" (TASK_MODEL_RATIO.md:5)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for TRANSFORMER-NO from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was already sufficient for a high-confidence binary decision.
