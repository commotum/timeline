# A Neural Transducer (Not specified in the paper.)
Source: A Neural Transducer.md

## Core reasons
- The paper identifies a missing capability in standard sequence-to-sequence models: they cannot make incremental/online predictions as input arrives.
- It proposes a new transducer mechanism that emits variable-length outputs per input block to enable online sequence transduction.

## Evidence extracts
- "However, they are unsuitable for tasks that require incremental predictions to be made as more data arrives or tasks that have long input sequences and output sequences. This is because they generate an output sequence conditioned on an entire input sequence. In this paper, we present a Neural Transducer that can make incremental predictions as more input arrives, without redoing the entire computation." (Abstract)
- "Neural Transducer can produce chunks of outputs (possibly of zero length) as blocks of inputs arrive - thus satisfying the condition of being \"online\"" (1 Introduction)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
