# Addressing the Rare Word Problem in Neural Machine Translation (Not specified in the paper.)
Source: Addressing the Rare Word Problem in Neural Machine Translation.md

## Core reasons
- The paper proposes an alignment-augmented NMT training and post-processing method to handle OOV words, which is a modeling technique rather than a dataset or positional encoding innovation.
- The contribution centers on improving translation behavior in NMT systems through a rare-word handling mechanism, fitting a general ML methods category.

## Evidence extracts
- "In this paper, we propose and implement an effective technique to address this problem. We train an NMT system on data that is augmented by the output of a word alignment algorithm, allowing the NMT system to emit, for each OOV word in the target sentence, the position of its corresponding word in the source sentence." (Abstract)
- "We have shown that a simple alignment-based technique can mitigate and even overcome one of the main weaknesses of current NMT systems, which is their inability to translate words that are not in their vocabulary." (Section 6 Conclusion)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
