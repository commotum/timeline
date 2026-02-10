# Walrus: A Cross-domain Foundation Model for Continuum Dynamics (Not specified in the paper.)
Source: WALRUS- A Cross-Domain Foundation Model for Continuum Dynamics.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Continuum dynamics emulation / next-step prediction | Short sequence of continuum state snapshots across physical systems | 3D (x, y, t); 4D (x, y, z, t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Next-step state increment and next system state (used autoregressively for rollouts) | 2D (x, y); 3D (x, y, z) (inferred) | Capped (inferred) |

## Summary
Walrus is presented as a cross-domain foundation model for continuum dynamics emulation, and the paper consistently frames its task as next-step prediction from short state-history sequences. The reported coverage spans both 2D and 3D physical systems, which maps to spatiotemporal input domains of 3D (x, y, t) and 4D (x, y, z, t) and spatial outputs in 2D/3D. Based on the described fixed history windows and standard causal/self-attention over provided tokens, the justified classifications are Capped dynamics with Static attention and Direct state handling.

## Evidence
### Task: Continuum dynamics emulation / next-step prediction
- "Our interest is in data-driven emulation of physical systems, specifically at the level of continuum operators." (Section 2 Background)
- "Walrus takes as input a short sequence of snapshots and predicts the next step in the sequence." (Figure 1)
- "All models are trained or finetuned to predict the next system state" (Section 5 Experiments, Training settings)
- "Importantly, we use both 2D and 3D data during pretraining." (Section 5 Experiments, Data)
- Inference: In Dimension is mapped to "3D (x, y, t); 4D (x, y, z, t)" from the paper's sequence-over-time formulation and mixed 2D/3D coverage ("given a sequence of tau snapshots" in Section 2; "both 2D and 3D data" in Sections 1 and 5). In/Out Dynamics are marked "Capped" from explicit fixed history settings ("Time History (2D) 6" and "Time History (3D) 3" in Table 2) plus variable-but-bounded task configurations. Attention is "Static" and State is "Direct" because Walrus applies causal/self-attention over a provided history window rather than runtime retrieval/action selection or persistent external state construction (Section 3.1; Section A.3).
