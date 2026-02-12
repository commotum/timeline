# Improving Image Generation with Better Captions (Year not specified)
Source: Improving Image Generation with Better Captions.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The central text-to-image system is a hybrid that explicitly uses a Transformer-family text encoder (`T5`) as a core conditioning component for DALL-E 3.
- The model is paired with a diffusion U-Net decoder (hybrid architecture), and the Extending-dimensions analysis file was unavailable (`MISSING`), so the decision relies on abstract + available auxiliary files + targeted scan.

## Evidence
- "We suspect this may have to do with the T5 text encoder we used: when the model encounters text in a prompt, it" (Section 5.2 Text rendering, Improving Image Generation with Better Captions.md)
- "The image decoder used in our experiments is a text-conditioned U-Net[23] latent diffusion model[22] with three stages." (Appendix A Image decoder, Improving Image Generation with Better Captions.md)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Abstract and available auxiliary files were read in full; they established the DALL-E 3 + captioner pipeline, while the Extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): performed - Needed for explicit architecture cues; scan found direct use of a "T5 text encoder" in the deployed system.
