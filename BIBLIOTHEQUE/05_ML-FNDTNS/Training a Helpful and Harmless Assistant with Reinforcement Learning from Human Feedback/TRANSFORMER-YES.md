# Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback (2022)
Source: Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: medium
Basis: source-targeted-scan

## Why
- The paper's main method is RLHF over large autoregressive language models (up to 52B), and those base models are the central systems used for all primary results.
- The extending-dimensions analysis file was unavailable (`MISSING`), so the decision relies on the abstract, available auxiliary files, and targeted architecture-cue lines from the source.

## Evidence
- "We apply preference modeling and reinforcement learning from human feedback (RLHF) to finetune language models" (Abstract, `Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.md:16`)
- "used 52B language models with the broad specifications given in [Askell et al., 2021]" (Section 2.3, `Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.md:208`)
- "train an RL policy to generate a response to each prompt autoregressively" (Section 4.1, `Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.md:330`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Indicated a large-language-model RLHF core, but self-attention architecture was only implicit.
Pass 2 (targeted source scan): performed - Confirmed central use of 52B autoregressive language models and RL policy generation, supporting Transformer-family classification.
