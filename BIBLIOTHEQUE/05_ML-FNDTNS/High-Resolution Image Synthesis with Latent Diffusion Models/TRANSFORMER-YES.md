# High-Resolution Image Synthesis with Latent Diffusion Models (Year not specified)
Source: High-Resolution Image Synthesis with Latent Diffusion Models.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract states that cross-attention layers are introduced directly into the model architecture, indicating transformer-style attention is part of the core method.
- The auxiliary model-ratio analysis quotes the paper’s contribution that a general-purpose conditioning mechanism is based on cross-attention and is used for major reported tasks.
- The Extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files already provide clear architecture evidence.

## Evidence
- "By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes..." (Abstract, `High-Resolution Image Synthesis with Latent Diffusion Models.md`)
- "(v) Moreover, we design a general-purpose conditioning mechanism based on cross-attention, enabling multi-modal training. We use it to train class-conditional, text-to-image and layout-to-image models." (Section 1, Contributions, `TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for a high-confidence YES decision using abstract + `TASK-DOMAINS.md` + `TASK-DOMAINS.csv` + `TASK_MODEL_RATIO.md`; Extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was already sufficient.
