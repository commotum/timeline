# Scaling Instructable Agents Across Many Simulated Worlds (SIMA) (Year not specified)
Source: Scaling Instructable Agents Across Many Simulated Worlds (SIMA).md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The auxiliary analysis explicitly states the main SIMA agent uses trained-from-scratch Transformers with cross-attention and Transformer-XL memory, indicating self-attention is central to the core model.
- The abstract defines a single language-and-vision-to-action embodied agent, and TASK_MODEL_RATIO indicates one model is used across tasks, so the Transformer-based architecture is material to the paper’s main results.

## Evidence
- "Building embodied AI systems that can follow arbitrary language instructions in any 3D environment is a key challenge for creating general AI." (Scaling Instructable Agents Across Many Simulated Worlds (SIMA).md, abstract paragraph)
- "our agent (Figure 4) utilizes trained-from-scratch transformers that cross-attend to the different pretrained vision components, the encoded language instruction, and a Transformer-XL (Dai et al., 2019) that attends to past memory states to construct a state representation." (TASK-DOMAINS.md, Evidence section quoting Section 3.3 Agent)
- "The SIMA agent maps visual observations and language instructions to keyboard-and-mouse actions (Figure 4)." (TASK-DOMAINS.md, Evidence section quoting Section 3.3 Agent)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for high-confidence decision; Extending-dimensions analysis markdown was unavailable (MISSING) and skipped.
Pass 2 (targeted source scan): skipped - Pass 1 already gave explicit Transformer/self-attention architecture evidence.
