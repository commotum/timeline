# The Power of Scale for Parameter-Efficient Prompt Tuning (2021)
Source: The Power of Scale for Parameter-Efficient Prompt Tuning.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The paper's core method adapts frozen T5 models, and the method section explicitly describes these as Transformer encoder-decoder components.
- The main experiments and scaling results are run on these T5 models, so Transformer architecture is central to the paper's primary results.

## Evidence
- "T5 models classification as  $\Pr_{\theta}(Y|X)$ , parameterized by the weights,  $\theta$ , of the transformers (Vaswani et al., 2017) that make up its encoder and decoder." (Section 2 Prompt Tuning, The Power of Scale for Parameter-Efficient Prompt Tuning.md:52)
- "through ablations on model size using T5, we show that prompt tuning becomes more competitive with scale" (Abstract, The Power of Scale for Parameter-Efficient Prompt Tuning.md:11)
- "The paper covers multiple NLP task domains using frozen T5 models with prompt tuning" (Summary, TASK-DOMAINS.md:16)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Read abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md in full; Extending-dimensions analysis file was unavailable (MISSING).
Pass 2 (targeted source scan): performed - Needed explicit architecture confirmation; Section 2 directly states the model uses transformers in an encoder-decoder setup.
