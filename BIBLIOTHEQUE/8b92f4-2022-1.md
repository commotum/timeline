# Tackling the Abstraction and Reasoning Corpus with Vision Transformers: The Importance of 2D Representation, Positions, and Objects (2025)
Source: 8b92f4-2022.pdf

## Core reasons
- The paper argues that transformers prioritize token embeddings over positional encodings for ARC tasks, limiting their ability to capture spatial relationships.
- The main architectural contribution is new positional-encoding mechanisms (notably object-based positional encoding plus 2D positional handling) to improve spatial reasoning in a ViT.

## Evidence extracts
- "tasks remain poorly solved. After further failure analysis on these tasks, we discover that certain complex visual structures are difficult for VITARC. We hypothesize this is due to limitations of the transformer architecture itself in that it is designed to prioritize token embeddings over positional encodings that can make it challenging to capture intricate spatial relationships." (p. 2)
- "Object-based Positional Encoding (OPE). For tasks involving multi-colored objects, or more generally, tasks that require objectness priors (Chollet, 2019), external sources of knowledge about object abstractions can be integrated into the model. We inject this information through a novel object-based positional encoding. We extend the 2D sinusoidal APE defined in Equation (9) by introducing the object index o as an additional component to the pixel coordinates (x,y). This results in a modified positional encoding:" (p. 9)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
