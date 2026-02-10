# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (2020)
Source: ZeRO- Memory Optimizations Toward Training Trillion Parameter Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling | Token sequences (text) (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Next-token probability predictions (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The OCR provides explicit evidence that ZeRO is applied to training large language models, specifically GPT-2-like transformer models and Turing-NLG. The task modality is text sequence modeling, which supports a 1D (t) characterization for both input and output as an inference from the language-model framing. Reported sequence lengths (1K/1024) support capped dynamics for the modeled sequence interface. The paper does not explicitly specify attention-policy type or whether decision state is direct versus constructed under the glossary definitions.

## Evidence
### Task: Language modeling
- "The models presented in this section are GPT-2 [2] like transformer based models." (Section 10.1)
- "As of May 12th, 2020, Turing-NLG is the largest model in the world with over 17B parameters. It achieved the new SOTA for language models with Webtext-103 perplexity of 10.21." (Section 10.6)
- "Activations can take up a significant amount of memory [7] during training. As a concrete example, the 1.5B parameter GPT-2 model trained with sequence length of 1K and batch size of 32 requires about 60 GB of memory<sup>3</sup>." (Section 3.2)
- "Memory Saving With partitioned activation checkpointing, ZeRO reduces the activation footprint by a factor proportional to the MP degree. Consider training a 100B model shown in Table 4 with a batch size of 32, sequence length of 1024 and a MP degree of 16." (Section 6.1)
- Inference: "Language models" with GPT-2-like transformer setups imply ordered token-sequence processing and next-token probability prediction, supporting `Input/Output = token sequences/predictions` and `Dimension = 1D (t)` (inferred). The explicit sequence-length settings ("sequence length of 1K" and "sequence length of 1024") justify `In Dynamics` and `Out Dynamics` as `Capped` (inferred). The OCR does not explicitly characterize runtime attention control or state-construction behavior in glossary terms, so those fields are marked "Not specified in the paper.".
