# Hierarchical Text-Conditional Image Generation with CLIP Latents (Year not specified)
Source: Hierarchical Text-Conditional Image Generation with CLIP Latents.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The paper’s core text-to-image stack uses a prior model, and the method section explicitly specifies Transformer architectures with causal attention masks for both AR and diffusion priors.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract + available auxiliary files plus targeted method lines provide direct architecture evidence sufficient for a high-confidence decision.

## Evidence
- "We use diffusion models for the decoder and experiment with both autoregressive and diffusion models for the prior..." (Hierarchical Text-Conditional Image Generation with CLIP Latents.md:17, Abstract)
- "predict the resulting sequence using a Transformer [53] model with a causal attention mask." (Hierarchical Text-Conditional Image Generation with CLIP Latents.md:113, Section 2.2 Prior)
- "For the diffusion prior, we train a decoder-only Transformer with a causal attention mask..." (Hierarchical Text-Conditional Image Generation with CLIP Latents.md:117, Section 2.2 Prior)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - showed AR/diffusion prior in the core stack, but was not alone fully explicit about Transformer internals.
Pass 2 (targeted source scan): performed - found direct statements that the priors are Transformer models with causal attention masks.
