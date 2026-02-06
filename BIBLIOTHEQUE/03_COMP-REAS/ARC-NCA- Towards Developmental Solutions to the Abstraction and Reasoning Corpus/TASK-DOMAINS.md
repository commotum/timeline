# ARC-NCA: Towards Developmental Solutions to the Abstraction and Reasoning Corpus (Not specified in the paper.)
Source: ARC-NCA- Towards Developmental Solutions to the Abstraction and Reasoning Corpus.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visual pattern transformation | 2D grids (input-output example pairs and test input grid) | 2D (x, y) | Capped | Static (inferred) | Constructed (inferred) | output grid | 2D (x, y) | Capped |

## Summary
The paper targets ARC-AGI visual pattern transformation tasks where systems infer rules from a few input-output grid examples and produce an output grid for a test input. Inputs and outputs are 2D grids with bounded sizes up to 30x30, so the task operates over 2D (x, y) with capped dynamics. The NCA/EngramNCA description implies fixed local processing (static attention) and internal cell memory (constructed state), though these properties are inferred from the architecture details.

## Evidence
### Task: visual pattern transformation
- "It comprises a collection of visual pattern transformation tasks, each defined by a few input-output examples, challenging AI models to infer the underlying transformation rules and apply them to novel instances." (Introduction)
- "Test pairs consist of two components: an \"input grid,\" which is a rectangular array of cells with varying dimensions (up to 30 rows by 30 columns), where each cell holds one of ten distinct \"values,\" and an \"output grid,\" which can be entirely derived from the attributes and structure of the input grid." (Introduction)
- "The purpose is to examine the example pairs to grasp the nature of the problem and utilize this understanding to produce the corresponding output grid for each given test input." (Introduction)
- "The ARC dataset mainly comprises 2D grids with integer values. Each grid can range from 1x1 to 30x30 in size, with values ranging between 0 and 9." (From ARC to NCA Space)
- Inference: Attention Dynamic marked Static because the model applies fixed local processing ("local update rules") and local self-attention over predefined neighborhoods, matching a predefined slice of input. This is inferred from: "each cell maintains a continuous state vector updated through convolutional neural networks (CNNs) with learned local update rules" (NCA models) and "giving each cell channel-wise local self-attention." (ARC Specific Augmentations)
- Inference: State Dynamic marked Constructed because cells maintain internal state and EngramNCA includes private memory states beyond the raw grid input. This is inferred from: "each cell maintains a continuous state vector" (NCA models) and "dual-state cells with distinct public (interaction-based) and private (memory-based) states." (NCA models)
