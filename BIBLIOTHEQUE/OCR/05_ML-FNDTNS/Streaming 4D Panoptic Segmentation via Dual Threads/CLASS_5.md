# 4DSegStreamer: Streaming 4D Panoptic Segmentation via Dual Threads (Not specified in the paper.)
Source: Streaming 4D Panoptic Segmentation via Dual Threads.md

## Core reasons
- Proposes a new real-time streaming 4D panoptic segmentation method with a dual-thread system and memory/forecasting mechanics, which is a modeling/architecture contribution rather than a dataset or benchmark.
- Focuses on system design for streaming inference (predictive and inference threads, memory update, motion alignment), not on positional encoding or transformer dimensional lifting as the primary contribution.

## Evidence extracts
- "In this paper, we introduce 4DSegStreamer, a novel framework that employs a Dual-Thread System to efficiently process streaming frames." (Abstract)
- "We propose a new task of streaming 4D panoptic segmentation." (Section 3)
- "4DSegStreamer employs a novel dual-thread system comprising a predictive thread and an inference thread, which is general and can be applied to various segmentation methods to enable their real-time performance." (Section 4)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
