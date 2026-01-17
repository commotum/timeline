# DEBERTA: DECODING-ENHANCED BERT WITH DIS-ENTANGLED ATTENTION (Not specified in the paper.)
Source: DeBERTa- Decoding-enhanced BERT with Disentangled Attention.md

## Core reasons
- The core contribution modifies positional handling by disentangling content and position vectors and computing attention with relative positions.
- It further changes positional encoding usage by adding absolute position information in the decoding layer via an enhanced mask decoder.

## Evidence extracts
- "we propose a new model architecture DeBERTa (Decoding-enhanced BERT with disentangled attention) that improves the BERT and RoBERTa models using two novel techniques. The first is the disentangled attention mechanism, where each word is represented using two vectors that encode its content and position, respectively, and the attention weights among words are computed using disentangled matrices on their contents and relative positions, respectively. Second, an enhanced mask decoder is used to incorporate absolute positions in the decoding layer to predict the masked tokens in model pre-training." (Abstract)
- "The standard self-attention mechanism lacks a natural way to encode word position information. Thus, existing approaches add a positional bias to each input word embedding" (Section 2.1 Transformer)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
