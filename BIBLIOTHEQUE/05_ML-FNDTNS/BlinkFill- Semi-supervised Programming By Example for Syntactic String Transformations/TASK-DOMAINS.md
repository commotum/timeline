# BlinkFill: Semi-supervised Programming By Example for Syntactic String Transformations (2016)
Source: BlinkFill- Semi-supervised Programming By Example for Syntactic String Transformations.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| String transformation (substring extraction and concatenation) | Spreadsheet row strings (one or more input strings/columns) | 1D (t) | Open (inferred) | Static (inferred) | Constructed (inferred) | Transformed strings (concatenated substrings/constant strings) | 1D (t) | Open (inferred) |

## Summary
BlinkFill targets syntactic string transformation tasks in spreadsheets, producing output strings formed by substring extraction and concatenation. Inputs and outputs are 1D character sequences, and the transformations operate over variable-length strings (inferred from "varying length" examples, with no explicit maximum given). The paper does not define an attention mechanism, but the described execution semantics are fixed by the learned program (inferred static attention) and rely on constructed internal structures and a learned DSL program (inferred constructed state).

## Evidence
### Task: String transformation (substring extraction and concatenation)
- "the string transformation task involves transforming a set of n input row strings" (Section 2. PRELIMINARIES)
- "The top-level string expression e is a concatenation of a finite list of substring expressions" (Section 6.1 String Transformation Language)
- "A string s is considered as simply a sequence of characters" (Section 2. PRELIMINARIES)
- Inference: In/Out Dynamics = Open (inferred) because outputs can be "of varying length" and no explicit maximum length is specified. "extract the information (of varying length)" (Section 3. MOTIVATING EXAMPLES)
- Inference: Attention Dynamic = Static (inferred) because evaluation follows a fixed concatenation of substring expressions. "The semantics of a concatenate expression is to recursively evaluate each individual substring expression f_i and then concatenate them." (Section 6.1 String Transformation Language)
- Inference: State Dynamic = Constructed (inferred) because the system learns and reuses a program. "learn a program in the DSL that is consistent with the input-output examples, and executes it on the spreadsheet data" (Section 4. OVERVIEW OF THE APPROACH)
