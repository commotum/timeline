# CAUSALEMBED: Auto-Regressive Multi-Vector Generation in Latent Space for Visual Document Embedding (2026)
Source: CausalEmbed- Auto-Regressive Multi-Vector Generation in Latent Space for Visual Document Embedding.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The paper’s core method is built around Multimodal Large Language Models and autoregressive embedding generation, indicating a Transformer-family backbone is central rather than peripheral.
- The auxiliary analysis explicitly reports CAUSALEMBED experiments on PaliGemma-3B and Qwen2.5-VL-3B backbones, both established Transformer-based VLM/MLLM architectures.
- The Extending-dimensions analysis markdown was unavailable (`MISSING`), but the abstract plus available auxiliary files already provide sufficient architecture signals.

## Evidence
- "Although Multimodal Large Language Models (MLLMs) have shown remarkable potential in Visual Document Retrieval (VDR) ..." (CausalEmbed- Auto-Regressive Multi-Vector Generation in Latent Space for Visual Document Embedding.md, Abstract, line 7)
- "To evaluate the effectiveness of CAUSALEMBED, we conduct experiments using various backbones of different sizes, including PaliGemma-3B ... and Qwen2.5-VL-3B ..." (TASK_MODEL_RATIO.md, line 9)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for high-confidence decision using abstract, `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; Extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided enough evidence to finalize.
