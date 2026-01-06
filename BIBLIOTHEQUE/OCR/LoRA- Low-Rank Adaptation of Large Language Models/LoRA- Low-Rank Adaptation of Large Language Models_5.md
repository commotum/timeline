# LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS (Not specified in the paper.)
Source: LoRA- Low-Rank Adaptation of Large Language Models.md

## Core reasons
- The paper proposes a parameter-efficient fine-tuning method that freezes pretrained weights and trains low-rank update matrices, which is a training/optimization contribution rather than positional encoding or dimensional adaptation.
- The main contribution is a low-rank reparameterization of weight updates for Transformer layers to reduce trainable parameters and memory, aligning with ML foundations/principles on efficient adaptation.

## Evidence extracts
- "We propose Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks." (Abstract)
- "For a pre-trained weight matrix  $W_0 \in \mathbb{R}^{d \times k}$ , we constrain its update by representing the latter with a low-rank decomposition  $W_0 + \Delta W = W_0 + BA$ , where  $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$ , and the rank  $r \ll \min(d, k)$ ." (Section 4.1 LOW-RANK-PARAMETRIZED UPDATE MATRICES)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
