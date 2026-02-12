# Consistency Models (2023)
Source: Consistency Models.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: source-targeted-scan

## Why
- The abstract and auxiliary analyses focus on consistency/diffusion objectives and downstream image tasks, without identifying a Transformer-family architecture as central.
- The paper’s architecture details point to diffusion-model backbones (NCSN++ / EDM-linked architectures) rather than Transformer blocks as the main model family.

## Evidence
- "Specifically, we use the NCSN++ architecture in Song et al. (2021) for all CIFAR-10 experiments, and take the corresponding network architectures from Dhariwal & Nichol (2021) when performing experiments on ImageNet  $64 \times 64$ , LSUN Bedroom  $256 \times 256$  and LSUN Cat  $256 \times 256$ ." (Consistency Models.md, Appendix C, Model Architectures)
- "Importantly, neither approach necessitates adversarial training, and they both place minor constraints on the architecture, allowing the use of flexible neural networks for parameterizing consistency models." (Consistency Models.md, Introduction)
- "Not specified in the paper." (TASK-DOMAINS.csv, `attention_dynamic` field across all listed tasks)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Abstract, TASK-DOMAINS.md, TASK-DOMAINS.csv, and TASK_MODEL_RATIO.md were read in full; extending-dimensions analysis file was unavailable (MISSING).
Pass 2 (targeted source scan): performed - Needed architecture-specific cues; targeted scan found explicit model architecture statements (Appendix C).
