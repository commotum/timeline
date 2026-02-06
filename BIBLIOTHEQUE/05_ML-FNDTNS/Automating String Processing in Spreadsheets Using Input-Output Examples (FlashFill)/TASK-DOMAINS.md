# Automating String Processing in Spreadsheets Using Input-Output Examples (2011)
Source: Automating String Processing in Spreadsheets Using Input-Output Examples (FlashFill).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| program synthesis (string manipulation) | input-output examples of spreadsheet strings (multiple input columns) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | string processing program / string expression | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper presents a program synthesis system that learns string processing programs for spreadsheet data from input-output examples. The task domain centers on string manipulation in spreadsheets, with inputs described as strings in spreadsheet columns and outputs as synthesized string programs. The text supports a 1D string-based input dimension and constructed internal state, while dynamics and attention are not explicitly specified.

## Evidence
### Task: program synthesis (string manipulation)
- "We describe an algorithm based on several novel concepts for synthesizing a desired program in this language from input-output examples." (Abstract)
- "We describe a program synthesis system that is capable of synthesizing a wide range of string processing programs in spreadsheets from input-output examples." (Introduction)
- "input state σ, which holds values for m string variables v1, ..., vm (denoting the multiple input columns in a spreadsheet)" (Section 3)
- "a key enabling technology is the data-structure (described in Section 4.1) for succinctly representing and manipulating such a huge set of expressions." (Section 4)
- Inference: In Dimension labeled 1D (t) because the inputs are strings in spreadsheet columns; State Dynamic marked Constructed because the algorithm represents and manipulates large sets of expressions via a dedicated data-structure (quotes above).
