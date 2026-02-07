# Keeping Neural Networks Simple by Minimizing the Description Length of the Weights (Not specified in the paper.)
Source: Keeping Neural Networks Simple by Minimizing the Description Length of the Weights.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prediction of peptide effectiveness | peptide molecule parameter vector (128 parameters) | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | effectiveness scalar | 0D (inferred) | Fixed (inferred) |

## Summary
The paper reports a single supervised prediction task on peptide molecules, using fixed-length input vectors to predict a scalar effectiveness value. This supports 1D (t) fixed inputs and 0D fixed outputs, while attention and state dynamics are not specified.

## Evidence
### Task: prediction of peptide effectiveness
- "The task is to predict the effectiveness of a class of peptide molecules." (Section 9 Preliminary Results)
- "Each molecule is described by 128 parameters (the input vector) and has an effectiveness that is a single scalar (the ouput value)." (Section 9 Preliminary Results)
- Inference: Classified input as 1D (t) and Fixed and output as 0D and Fixed because the task uses a 128-parameter input vector and a single scalar output. (Section 9 Preliminary Results)
