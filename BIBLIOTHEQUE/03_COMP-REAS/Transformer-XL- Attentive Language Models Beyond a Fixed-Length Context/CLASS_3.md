# Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (Not specified in the paper.)
Source: Transformer-XL- Attentive Language Models Beyond a Fixed-Length Context.md

## Core reasons
- The paper frames fixed-length context as a limitation for Transformers and proposes an architecture that enables learning dependencies beyond that limit.
- The central mechanism is segment-level recurrence with state reuse, changing computation by caching and reusing past hidden states as memory across segments.

## Evidence extracts
- "Transformers have a potential of learning longer-term dependency, but are limited by a fixed-length context in the setting of language modeling. We propose a novel neural architecture Transformer-XL that enables learning dependency beyond a fixed length without disrupting temporal coherence." (Abstract)
- "To address the limitations of using a fixed-length context, we propose to introduce a recurrence mechanism to the Transformer architecture. During training, the hidden state sequence computed for the previous segment is fixed and cached to be reused as an extended context when the model processes the next new segment, as shown in Fig. 2a." (Section 3.2 Segment-Level Recurrence with State Reuse)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
