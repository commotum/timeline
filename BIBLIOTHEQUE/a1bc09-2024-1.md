# DAPE: Data-Adaptive Positional Encoding for Length Extrapolation (2024)
Source: a1bc09-2024.pdf

## Core reasons
- Static absolute and relative positional encodings are criticized for being fixed after training, so the paper proposes DAPE to dynamically adjust positional information based on attention context (Abstract).
- Section 3 explains that DAPE builds on additive relative positional encodings by using semantically adaptive bias matrices computed with MLPs, highlighting that the core innovation alters positional encoding mechanics (Section 3).

## Evidence extracts
- "Positional encoding plays a crucial role in transformers, significantly impacting model performance and length generalization. Prior research has introduced absolute positional encoding (APE) and relative positional encoding (RPE) to distinguish token positions in given sequences. However, both APE and RPE remain fixed after model training regardless of input data, limiting their adaptability and flexibility. Hence, we expect that the desired positional encoding should be data-adaptive and can be dynamically adjusted with the given attention. In this paper, we propose a Data-Adaptive Positional Encoding (DAPE) method, which dynamically and semantically adjusts based on input context and learned fixed priors." (Abstract)
- "In this section, we formally introduce DAPE (data-adaptive positional encoding), a new relative positional encoding approach that further enhances transformer performance. Compared with previous works on static positional encoding methods, DAPE adopts semantically adaptive positional bias matrices depending on input context. We first demonstrate that most of the popular positional bias matrices are fixed once the training is finished, independent of the input sequences. To address this limitation, we then accordingly develop DAPE that captures the implicit relationships by MLPs and adjusts the bias matrices based on input context." (Section 3)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
