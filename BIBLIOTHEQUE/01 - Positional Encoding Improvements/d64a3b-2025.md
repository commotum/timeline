# Wavelet-Based Positional Representation for Long Context (2025)
Source: d64a3b-2025.pdf

## Core reasons
- The paper critiques existing positional encodings (e.g., RoPE’s fixed scale limits extrapolation), highlighting limitations for long-context handling.
- It proposes a wavelet transform-based positional representation to capture multiple scales for positional encoding.

## Evidence extracts
- "                regarded as the time axis, Rotary Position Embedding (RoPE) can be interpreted
                as a restricted wavelet transform using Haar-like wavelets. However, because
                it uses only a fixed scale parameter, it does not fully exploit the advantages of
                wavelet transforms, which capture the fine movements of non-stationary signals
                using multiple scales (window sizes). This limitation could explain why RoPE
                performs poorly in extrapolation." (Abstract)
- "                           Based on these insights, we propose a wavelet transform-based method, using multiple window
                           sizes, to offer a robust and flexible approach to positional encoding." (Introduction)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
