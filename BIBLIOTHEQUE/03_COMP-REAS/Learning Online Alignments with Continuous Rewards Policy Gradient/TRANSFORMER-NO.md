# Learning Online Alignments with Continuous Rewards Policy Gradient (Year not specified)
Source: Learning Online Alignments with Continuous Rewards Policy Gradient.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract and method describe a recurrent LSTM sequence-to-sequence model with hard stochastic emission decisions trained by policy gradient, not Transformer blocks or self-attention.
- Attention is discussed as prior/offline soft attention context, while the proposed method centers on online hard alignments in an RNN; the Extending-dimensions analysis markdown input was unavailable (MISSING).

## Evidence
- "Our model uses hard binary stochastic decisions to select the timesteps at which outputs will be produced." (Abstract, Learning Online Alignments with Continuous Rewards Policy Gradient.md)
- "h_i = LSTM(h_{i-1}, concat(x_i, \tilde{b}_{i-1}, \tilde{y}_{i-1}))" (Section 2 Methods quote recorded in TASK-DOMAINS.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for high-confidence NO decision; no central Transformer/self-attention architecture signal in abstract or auxiliary analyses.
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient; Extending-dimensions analysis markdown was unavailable (MISSING).
