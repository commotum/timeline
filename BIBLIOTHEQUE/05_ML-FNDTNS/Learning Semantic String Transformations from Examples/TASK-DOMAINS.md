# Learning Semantic String Transformations from Examples (2012)
Source: Learning Semantic String Transformations from Examples.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Semantic string transformation | strings (spreadsheet column values); relational tables (lookup tables) | 1D (t); 2D (x, y) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | string (transformed string) | 1D (t) | Not specified in the paper. |

## Summary
The paper targets semantic string transformations in spreadsheets, combining table lookup and syntactic string manipulation learned from input-output examples. Inputs are strings (1D) alongside relational tables (2D), and outputs are transformed strings (1D). The paper does not specify interface dynamics or attention/state behavior.

## Evidence
### Task: Semantic string transformation
- "We address the problem of performing semantic transformations on strings" (Abstract)
- "mapping a tuple of strings to another string using (possibly nested) lookup operations over a given database of relational tables." (§4 Lookup Transformations)
- "Select(C,T,b) returns the table entry T[C,r], where r is the only row that satisfies condition b" (§4.1 Lookup Transformation Language)
- "interpret a string as a sequence of characters" (Abstract)
