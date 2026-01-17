# ROTHP: ROTARY POSITION EMBEDDING-BASED TRANSFORMER HAWKES PROCESS (Not specified in the paper.)
Source: RoTHP- Rotary Position Embedding-based Transformer Hawkes Process.md

## Core reasons
- The paper critiques existing THP positional encodings as sensitive to timestamp translations and proposes a rotary positional embedding to address that limitation.
- The main contribution is an adaptation of positional encoding within a transformer-based Hawkes process, emphasizing translation invariance via relative time embeddings.

## Evidence extracts
- "Conventional THP and its variants simply adopt initial sinusoid embedding in transformers, which shows performance sensitivity to temporal change or noise in sequence data analysis by our empirical study. To deal with the problems, we propose a new Rotary Position Embedding-based THP (RoTHP) architecture in this paper." (Abstract)
- "Consequently, for the current timestamp encoding issue, we turn our attention to the relative positional encoding method... Specifically, we focus on the Rotary Positional Encoding (RoPE). Our aim is to adapt the RoPE for use in the context of neural Hawkes Point Processes." (Section 2.2.1 Timestamp noise sensitivity)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
