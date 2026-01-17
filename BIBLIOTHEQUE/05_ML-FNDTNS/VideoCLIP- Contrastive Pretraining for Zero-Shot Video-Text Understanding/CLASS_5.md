# VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding (Not specified in the paper.)
Source: VideoCLIP- Contrastive Pretraining for Zero-Shot Video-Text Understanding.md

## Core reasons
- The paper's main contribution is a contrastive pre-training method for zero-shot video-text understanding, not a new positional encoding, dimensional lifting, or dataset.
- It introduces training techniques for better video-text alignment and harder negatives (overlapped positives and retrieval-augmented negatives), which are optimization/training methodology contributions.

## Evidence extracts
- "We present VideoCLIP, a contrastive approach to pre-train a unified model for zeroshot video and text understanding, without using any labels on downstream tasks. VideoCLIP trains a transformer for video and text by contrasting temporally overlapping positive video-text pairs with hard negatives from nearest neighbor retrieval." (Abstract)
- "We present VideoCLIP that aims to pre-train a *unified* video-text representation with contrastive learning using two key techniques (see Fig. 1) to compute the training objective." (Section 1 Introduction)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
