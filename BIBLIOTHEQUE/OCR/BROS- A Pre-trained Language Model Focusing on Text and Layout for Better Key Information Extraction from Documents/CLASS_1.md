# BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents (Not specified in the paper)
Source: BROS- A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents.md

## Core reasons
- The paper explicitly critiques absolute 2D positional encoding and replaces it with a relative-position spatial encoding to better model document layout.
- The central contribution is a new spatial encoding method integrated into Transformer attention for 2D text blocks, making positional encoding the primary innovation.

## Evidence extracts
- "we propose a pre-trained language model, named BROS (BERT Relying On Spatiality), that encodes relative positions of texts in 2D space and learns from unlabeled documents with areamasking strategy." (Abstract)
- "LayoutLM (Xu et al. 2020) simply encodes absolute x- and y-axis positions to each text blocks but the specific-point encoding is not robust on the minor position changes of text blocks. Instead, BROS employs relative positions between text blocks to explicitly encode spatial relations." (Encoding Spatial Information into BERT)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
