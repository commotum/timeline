# GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models (2021)
Source: GLIDE- Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The core text-conditioned generator explicitly uses a Transformer to encode text prompts, and that conditioning is injected throughout the diffusion model.
- The architecture routes token embeddings into attention layers at each layer of the main model, making attention-based conditioning central rather than peripheral.
- The extending-dimensions analysis markdown was unavailable (`MISSING`), so the decision used the abstract, available auxiliary files, and a minimal targeted source scan.

## Evidence
- "To condition on the text, we first encode it into a sequence of K tokens, and feed these tokens into a Transformer model (Vaswani et al., 2017)." (Section 4.1, source file line 188)
- "the last layer of token embeddings (a sequence of K feature vectors) is separately projected to the dimensionality of each attention layer throughout the ADM model, and then concatenated to the attention context at each layer." (Section 4.1, source file line 188)
- "First, we train a 3.5 billion parameter diffusion model that uses a text encoder to condition on natural language descriptions." (Introduction, source file line 21)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Read abstract, `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md` fully; evidence strongly suggested attention-based text conditioning, while extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): performed - Minimal scan of Section 4.1 confirmed explicit Transformer usage and attention-layer integration in the central model.
