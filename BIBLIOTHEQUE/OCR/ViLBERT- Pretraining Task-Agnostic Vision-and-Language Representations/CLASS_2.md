# ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks (Not specified in the paper.)
Source: ViLBERT- Pretraining Task-Agnostic Vision-and-Language Representations.md

## Core reasons
- The paper's main contribution is an architectural extension of BERT into a multi-modal two-stream transformer that jointly processes visual and textual inputs via co-attentional layers.
- It explicitly adapts transformer processing to visual inputs by operating over image regions alongside text, enabling vision-and-language modeling beyond 1D language sequences.

## Evidence extracts
- "We extend the popular BERT architecture to a multi-modal two-stream model, processing both visual and textual inputs in separate streams that interact through co-attentional transformer layers." (Abstract)
- "Our model which we call ViLBERT is shown in Fig. 1 and consists of two parallel BERT-style models operating over image regions and text segments." (Section 2.2)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
