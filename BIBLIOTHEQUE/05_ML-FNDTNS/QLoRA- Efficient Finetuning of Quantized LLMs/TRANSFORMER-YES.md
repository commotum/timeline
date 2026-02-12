# QLoRA: Efficient Finetuning of Quantized LLMs (Year not specified)
Source: QLoRA- Efficient Finetuning of Quantized LLMs.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract states QLoRA finetunes pretrained LLM families including LLaMA and T5; these are standard Transformer model families where self-attention is central.
- Auxiliary analyses consistently frame evaluation/training around RoBERTa, T5, and LLaMA model setups, reinforcing that the main results are on Transformer-based architectures.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract plus available auxiliary files were sufficient for a high-confidence classification.

## Evidence
- "QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA)." (Abstract, `QLoRA- Efficient Finetuning of Quantized LLMs.md`)
- "We use QLORA to finetune more than 1,000 models, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5)..." (Abstract, `QLoRA- Efficient Finetuning of Quantized LLMs.md`)
- "Our evaluations include GLUE [58] with RoBERTa-large [38], Super-NaturalInstructions (TKInstruct) [61] with T5 [49], and 5-shot MMLU [24] after finetuning LLaMA..." (Section 4 quote captured in `TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - Sufficient evidence from abstract + `TASK-DOMAINS.md`, `TASK-DOMAINS.csv`, and `TASK_MODEL_RATIO.md`; extending-dimensions file unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 already provided high-confidence architecture identification.
