# Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (Not specified in the paper.)
Source: Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.md

## Core reasons
- The main contribution is a training-systems method: composing tensor, pipeline, and data parallelism with an interleaved pipeline schedule to improve throughput at scale.
- The paper focuses on efficient large-scale training on GPU clusters rather than new positional encoding, higher-dimensional modeling, computation mechanisms, or datasets.

## Evidence extracts
- "In this paper, we show how tensor, pipeline, and data parallelism can be composed to scale to thousands of GPUs. We propose a novel interleaved pipelining schedule that can improve throughput by 10+% with memory footprint comparable to existing approaches." (ABSTRACT)
- "In this paper, we have shown how PTD-P (inter-node pipeline parallelism, intra-node tensor parallelism, and data parallelism) can be composed to achieve high aggregate throughput (502 petaFLOP/s) while training large models with a trillion parameters." (7 DISCUSSION AND CONCLUSION)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
