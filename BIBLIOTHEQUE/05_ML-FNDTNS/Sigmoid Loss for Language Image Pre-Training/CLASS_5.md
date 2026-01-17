# Sigmoid Loss for Language Image Pre-Training (Not specified in the paper.)
Source: Sigmoid Loss for Language Image Pre-Training.md

## Core reasons
- Proposes a new sigmoid loss for language-image pre-training to replace softmax contrastive loss, which is a training-objective contribution rather than positional encoding or a dataset resource.
- Emphasizes efficiency and scaling behavior of the training loss (batch size, memory efficiency), aligning with ML training methodology and principles.

## Evidence extracts
- "We propose a simple pairwise Sigmoid loss for Language-Image Pre-training (SigLIP). Unlike standard contrastive learning with softmax normalization, the sigmoid loss operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization." (Abstract)
- "Instead of the softmax-based contrastive loss, we propose a simpler alternative that does not require computing global normalization factors." (Section 3.2. Sigmoid loss for language image pre-training)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
