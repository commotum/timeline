# Denoising Diffusion Probabilistic Models (2020)
Source: Denoising Diffusion Probabilistic Models.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The main reverse-process network used for results is explicitly a U-Net with self-attention blocks, so attention is part of the core model rather than only a baseline mention.
- The architecture conditions layers with Transformer sinusoidal position embeddings, reinforcing that Transformer-style components are materially used in the central model.
- The auxiliary file `EXTENDING-DIMENSIONS.md` was unavailable (`MISSING`), so the decision used the abstract, available auxiliary files, and targeted source architecture lines.

## Evidence
- "To represent the reverse process, we use a U-Net backbone similar to an unmasked PixelCNN++ [52, 48] with group normalization throughout [66]. Parameters are shared across time, which is specified to the network using the Transformer sinusoidal position embedding [60]. We use self-attention at the  $16 \times 16$  feature map resolution [63, 60]." (Section 4 Experiments, `Denoising Diffusion Probabilistic Models.md:160`)
- "All models have two convolutional residual blocks per resolution level and self-attention blocks at the  $16 \times 16$  resolution between the convolutional blocks [6]. Diffusion time t is specified by adding the Transformer sinusoidal position embedding [60] into each residual block." (Appendix B Experimental details, `Denoising Diffusion Probabilistic Models.md:406`)
- "We use a U-Net with self-attention; NCSN uses a RefineNet with dilated convolutions. We condition all layers on t by adding in the Transformer sinusoidal position embedding, rather than only in normalization layers (NCSNv1) or only at the output (v2)." (Appendix C Discussion on related work, `Denoising Diffusion Probabilistic Models.md:426`)
- "EXTENDING-DIMENSIONS.md missing" (Unavailable auxiliary input check)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Task/domain evidence was clear, but architecture-level attention centrality was not explicit enough for a high-confidence binary decision.
Pass 2 (targeted source scan): performed - Architecture sections explicitly confirmed self-attention and Transformer embedding use in the main model.
