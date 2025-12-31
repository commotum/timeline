# DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference (2020)
Source: 6fb87f-2020.pdf

## Core reasons
- Proposes dynamic early-exit off-ramps within BERT to let samples stop computation early, changing inference computation.
- Focuses on accelerating transformer inference via variable-depth execution rather than positional encoding or dimensional lifting.

## Evidence extracts
- "Our approach allows samples to exit earlier without passing through the entire model." (p. 1)
- "It adds one off-ramp for each transformer layer. An inference sample can exit earlier at an off-ramp, without going through the rest of the transformer layers." (p. 2)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
