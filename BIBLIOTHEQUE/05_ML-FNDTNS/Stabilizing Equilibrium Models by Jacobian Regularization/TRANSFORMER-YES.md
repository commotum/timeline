# Stabilizing Equilibrium Models by Jacobian Regularization (2021)
Source: Stabilizing Equilibrium Models by Jacobian Regularization.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s central method is evaluated with a DEQ-Transformer instantiation that explicitly uses multi-head self-attention as the underlying layer.
- The task/model auxiliary analysis shows a Transformer layer is one of the core trained model instances used for the paper’s main results (WikiText-103), so self-attention is materially part of the reported study.

## Evidence
- "Deep equilibrium networks (DEQs) are a new class of models that eschews traditional depth in favor of finding the fixed point of a single nonlinear layer." (Abstract, `Stabilizing Equilibrium Models by Jacobian Regularization.md`)
- "One of the very first successes of large-scale DEQs was its Transformer instantiation (Bai et al., 2019), which uses a multi-head self-attention (Vaswani et al., 2017) layer as the underlying  $f_{\theta}(\mathbf{z}; \mathbf{x})$  function." (`TASK_MODEL_RATIO.md`, quoted evidence from Section 5.2)
- "| A1-itt                             | 2-Layer ReLU block | Transformer layer              | Multiscale DEQ layer       | Multiscale DEQ layer      |" (`TASK_MODEL_RATIO.md`, Appendix A.3 Table 4 excerpt)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for a high-confidence decision; the extending-dimensions analysis file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient.
