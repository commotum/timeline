# ZEPHYR: DIRECT DISTILLATION OF LM ALIGNMENT (2023)
Source: 98bd11-2023.pdf

## Core reasons
- The paper centers on aligning a smaller open LLM purely through distilled preference learning (dSFT followed by dDPO) rather than architectural, encoding, or data-only contributions, making it fundamentally a machine-learning training method.
- The direct preference optimization procedure is the mechanism by which the model is refined and achieves state-of-the-art 7B performance, underscoring that the innovation is in the modeling/training foundation rather than positional or dimensional tricks.

## Evidence extracts
- "To validate this approach, we construct ZEPHYR-7B, an aligned version of Mistral-7B (Jiang et al., 2023). We first use dSFT, based on the UltraChat (Ding et al., 2023) dataset. Next we use the AI feedback data collected in the UltraFeedback dataset (Cui et al., 2023). Finally, we apply dDPO based on this feedback data." (p. 1)
- "Direct preference optimization (DPO) uses a simpler approach to directly optimize the preference model from the static data (Rafailov et al., 2023)." (p. 3)
- "Compared to other open 7B models, ZEPHYR-7B sets a new state-of-the-art and performs significantly better than dSFT models across both benchmarks." (p. 6)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
