# BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models (Not specified in the paper.)
Source: BLIP-2- Bootstrapping Language-Image Pre-training with Frozen Models.md

## Core reasons
- Proposes a new vision-language pre-training strategy that bootstraps from frozen image encoders and frozen large language models, which is a methodological/training contribution.
- Introduces a trainable Querying Transformer with a two-stage pre-training procedure to bridge the modality gap, emphasizing modeling and training design rather than data or positional encoding.

## Evidence extracts
- "This paper proposes BLIP-2, a generic and efficient pretraining strategy that bootstraps vision-language pre-training from off-the-shelf frozen pre-trained image encoders and frozen large language models." (Abstract)
- "We propose BLIP-2, a new vision-language pre-training method that bootstraps from frozen pre-trained unimodal models. In order to bridge the modality gap, we propose a Querying Transformer (Q-Former) pre-trained in two stages: (1) vision-language representation learning stage with a frozen image encoder and (2) vision-to-language generative learning stage with a frozen LLM." (Section 3. Method)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
