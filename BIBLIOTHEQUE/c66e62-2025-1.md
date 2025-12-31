# Maximizing the Position Embedding for Vision Transformers with Global Average Pooling (2025)
Source: c66e62-2025.pdf

## Core reasons
- The paper critiques a limitation in existing positional embedding usage in vision transformers, namely that PE is only added to token embeddings, limiting expressiveness.
- It proposes MPVG to maximize the effectiveness of PE in a layer-wise structure with GAP, a direct modification of how positional embedding is handled.

## Evidence extracts
- "In vision transformers, position embedding (PE) plays a cru- cial role in capturing the order of tokens. However, in vi- sion transformer structures, there is a limitation in the expres- siveness of PE due to the structure where position embed- ding is simply added to the token embedding." (p. 1)
- "In this paper, we identify the conflicting result that occurs in a layer-wise struc- ture when using the global average pooling (GAP) method instead of the class token. To overcome this problem, we pro- pose MPVG, which maximizes the effectiveness of PE in a layer-wise structure with GAP." (p. 1)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
