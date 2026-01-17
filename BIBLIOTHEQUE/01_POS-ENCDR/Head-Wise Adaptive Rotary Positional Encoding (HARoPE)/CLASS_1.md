# HEAD-WISE ADAPTIVE ROTARY POSITIONAL ENCODING FOR FINE-GRAINED IMAGE GENERATION (Not specified in the paper.)
Source: Head-Wise Adaptive Rotary Positional Encoding (HARoPE).md

## Core reasons
- The paper explicitly critiques standard multi-dimensional RoPE limitations (rigid frequency allocation, axis-wise independence, uniform head treatment) and targets positional encoding deficiencies.
- The main contribution is HARoPE, a head-wise adaptive modification to RoPE that changes the positional encoding mechanism before the rotary mapping.

## Evidence extracts
- "This paper identifies key limitations of standard multi-dimensional RoPE—rigid frequency allocation, axis-wise independence, and uniform head treatment—in capturing the complex structural biases required for fine-grained image generation. We propose HARoPE, a head-wise adaptive extension that inserts a learnable linear transformation parameterized via singular value decomposition (SVD) before the rotary mapping." (Abstract)
- "We propose HARoPE, a head-wise linear adaptation inserted immediately before the rotary mapping." (Section 3.3 Head-Wise Adaptive RoPE)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
