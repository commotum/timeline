# OptNet: Differentiable Optimization as a Layer in Neural Networks (2017)
Source: OptNet- Differentiable Optimization as a Layer in Neural Networks.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes OptNet as a differentiable quadratic-program optimization layer architecture, not a Transformer/self-attention architecture.
- Auxiliary analyses do not indicate Transformer-style self-attention as a central model component; reported model cues focus on fully connected and OptNet QP layers.
- The extending-dimensions analysis file was unavailable (`MISSING`), but abstract + available auxiliary evidence is sufficient for a high-confidence binary decision.

## Evidence
- "This paper presents OptNet, a network architecture that integrates optimization problems (here, specifically in the form of quadratic programs) as individual layers in larger end-to-end trainable deep networks." (Abstract, `OptNet- Differentiable Optimization as a Layer in Neural Networks.md`)
- "Attention and state dynamics are not explicitly characterized in the paper." (Summary, `TASK-DOMAINS.md`)
- "Specifically we use a FC600-FC10-SoftMax fully connected network and compare it to a FC600-FC10-Optnet10-SoftMax network, where the numbers after each layer indicate the layer size." (Quoted evidence block, `TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence for a high-confidence TRANSFORMER-NO decision from the abstract and auxiliary analyses.
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient; no additional source scan was needed.
