# QLORA:EfficientFinetuningofQuantizedLLMs (2023)
Source: 9756b7-2023.pdf

## Core reasons
- The paper introduces QLORA, an efficient fine-tuning method that lets a 65B parameter model be tuned on a single 48GB GPU while matching 16-bit performance, showing the focus is on training methodology rather than positional or dimensional modeling.
- It proposes quantization-specific mechanisms (NormalFloat quantization, Double Quantization, and Paged Optimizers) that keep gradients stable in low-bit training, reinforcing that the contribution targets ML foundations and optimization practices.

## Evidence extracts
- "We present QLORA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance." (p. 1)
- "QLORAachieveshigh-fidelity 4-bit finetuning via two techniques we propose—4-bit NormalFloat (NF4) quantization and Double Quantization. Additionally, we introduce Paged Optimizers, to prevent memory spikes during gradient checkpointing from causing out-of-memory errors that have traditionally made finetuning on a single machine difficult for large models." (p. 3)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
