# Understanding the RoPE Extensions of Long-Context LLMs (Year not specified)
Source: Understanding the RoPE Extensions of Long-Context LLMs.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract centers the work on LLM RoPE behavior through attention, which is a core Transformer mechanism.
- Auxiliary analyses frame the evaluated system as a decoder-only LLM with attention-based behavior across both tasks.
- The Extending-dimensions analysis markdown was unavailable (`MISSING`), but the abstract plus available auxiliary files were already sufficient for a high-confidence decision.

## Evidence
- "Most LLMs are built upon rotary position embedding (RoPE), a popular position encoding method." (Abstract, `Understanding the RoPE Extensions of Long-Context LLMs.md`)
- "This paper provides the first thorough understanding of RoPE extensions for long-context LLMs from an attention perspective, evaluated on two widely-used benchmarks: Perplexity and Needle-in-a-Haystack." (§5 quote captured in `TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence Transformer classification from abstract + `TASK-DOMAINS.md` + `TASK-DOMAINS.csv` + `TASK_MODEL_RATIO.md`; Extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - not needed after Pass 1.
