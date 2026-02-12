# Learning Iterative Reasoning through Energy Diffusion (2024)
Source: Learning Iterative Reasoning through Energy Diffusion.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: source-targeted-scan

## Why
- The central method is framed as energy-based iterative optimization rather than a Transformer/self-attention architecture.
- The paper’s explicit architecture descriptions use CNN, relational neural network, and STGCN-style graph convolution components, not Transformer blocks.
- The Extending-dimensions analysis markdown input was unavailable (`MISSING`), so this decision uses the abstract, provided auxiliary files, and a targeted architecture scan.

## Evidence
- "IRED learns energy functions to represent the constraints between input conditions and desired outputs." (Abstract, `Learning Iterative Reasoning through Energy Diffusion.md`:7)
- "It encodes the Sudoku board with a convolutional neural network with the residual connection design" (Appendix B, Discrete Task, `Learning Iterative Reasoning through Energy Diffusion.md`:442)
- "It uses a relational neural network to fuse the connectivity information from neighboring nodes." (Appendix B, Connectivity, `Learning Iterative Reasoning through Energy Diffusion.md`:444)
- "This is equivalent to the spatial-temporal graph convolution networks (STGCN; Yan et al., 2018)." (Appendix B, Planning Task, `Learning Iterative Reasoning through Energy Diffusion.md`:446)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Read abstract plus `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; these supported a non-Transformer framing but did not alone provide explicit architecture-family confirmation.
Pass 2 (targeted source scan): performed - Scanned architecture-focused sections in the paper markdown and found explicit non-Transformer architectures (CNN/relational network/STGCN) with no self-attention core.
