# Context-aware Biases for Length Extrapolation (2025)
Source: c0ab7b-2025.pdf

## Core reasons
- The paper identifies limitations of existing relative positional encoding biases for length extrapolation and motivates context-aware alternatives.
- The core contribution is a new additive relative positional encoding method (CABLE) that learns context-aware biases for Transformers.

## Evidence extracts
- "Most existing Relative Positional Encod-
ing (RPE) methods attempt to address this by
introducingeitherfixedlinearbiasesorglobally
learned biases, which lack the capacity to adapt
to different input contexts." (p. 1)
- "Weintroduced CABLE, a novel additive relative
positional encoding method that learns context-
aware biases for each token by injecting them
into the attention matrix at every decoder layer." (p. 9)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
