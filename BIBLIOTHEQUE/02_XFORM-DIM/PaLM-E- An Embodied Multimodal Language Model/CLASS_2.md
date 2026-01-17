# PaLM-E: An Embodied Multimodal Language Model (Not specified in the paper.)
Source: PaLM-E- An Embodied Multimodal Language Model.md

## Core reasons
- Proposes a transformer-based embodied language model that injects continuous sensor inputs (images, state estimates) into the LLM embedding space to handle non-text modalities.
- Central contribution is an architectural adaptation that converts 2D/3D/continuous observations into multimodal token sequences interleaved with text for Transformer processing.

## Evidence extracts
- "We propose embodied language models to directly incorporate real-world continuous sensor modalities into language models and thereby establish the link between words and percepts. Input to our embodied language model are multi-modal sentences that interleave visual, continuous state estimation, and textual input encodings." (Abstract)
- "The main architectural idea of PaLM-E is to inject continuous, embodied observations such as images, state estimates, or other sensor modalities into the language embedding space of a pre-trained language model." (3. PaLM-E: An Embodied Multimodal Language Model)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
