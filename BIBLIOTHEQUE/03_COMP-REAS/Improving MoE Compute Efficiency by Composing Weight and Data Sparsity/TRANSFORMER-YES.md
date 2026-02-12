# Improving MoE Compute Efficiency by Composing Weight and Data Sparsity (Year not specified)
Source: Improving MoE Compute Efficiency by Composing Weight and Data Sparsity.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The paper explicitly frames its MoE method as scaling Transformer architectures, with experts replacing/expanding Transformer FFN components under token routing.
- The core method is evaluated on upcycled Qwen3-based multimodal MoE models, consistent with a Transformer backbone; the extending-dimensions analysis file was unavailable (`MISSING`), so available files plus targeted source cues were used.

## Evidence
- "Mixture-of-Experts (MoE) layers [1; 2] have enabled more efficient scaling of Transformers through weight sparsity: they replicate the FFN into many experts and use a router to select a small subset per token" (Improving MoE Compute Efficiency by Composing Weight and Data Sparsity.md, Section 1 Introduction)
- "After warmup, we upcycle to MoE using a fine-grained expert shape inspired by [17]: 64 experts at  $4\times$  granularity, with 30% of FFN parameters reinitialized randomly. We upcycle from 0.6B and 1.7B Qwen3 [14] dense models" (Improving MoE Compute Efficiency by Composing Weight and Data Sparsity.md, Section 5.1 Training Details)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - likely Transformer-based MoE, but abstract+aux files alone did not explicitly establish self-attention centrality with high confidence; extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): performed - explicit Transformer/FFN MoE architecture statements found, enabling high-confidence YES.
