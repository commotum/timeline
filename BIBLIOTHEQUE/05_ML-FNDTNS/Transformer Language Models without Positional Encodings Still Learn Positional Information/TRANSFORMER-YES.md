# Transformer Language Models without Positional Encodings Still Learn Positional Information (2022)
Source: Transformer Language Models without Positional Encodings Still Learn Positional Information.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract identifies the central architecture as causal Transformer language models and analyzes their behavior without positional encodings.
- Auxiliary analyses (task/domain and task-model-ratio files) consistently describe the core experiments as Transformer/attention-based language modeling, including NoPos Transformer LMs and RoBERTa-based MLM.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files already provide sufficient architecture evidence.

## Evidence
- "Causal transformer language models (LMs), such as GPT-3, typically require some form of positional encoding, such as positional embeddings." (Abstract, Transformer Language Models without Positional Encodings Still Learn Positional Information.md)
- "Here, we demonstrate that transformer language models without any explicit positional information can and do learn an implicit notion of absolute positions that is sufficient to achieve competitive performance." (Introduction, Transformer Language Models without Positional Encodings Still Learn Positional Information.md)
- "To test our hypothesis, we run similar experiments for masked language models (MLM) (Devlin et al., 2019), which use order-invariant attention (since no causal mask is applied)." (Evidence block in TASK-DOMAINS.md, quoting Section 1 Introduction)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence TRANSFORMER-YES from the abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md; extending-dimensions analysis was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was already sufficient.
