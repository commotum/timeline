# AI Feynman: a Physics-Inspired Method for Symbolic Regression (2020)
Source: AI Feynman- a Physics-Inspired Method for Symbolic Regression.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| symbolic regression (discover symbolic expression from data) | table of numbers (rows: {x1, ..., xn, y}) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | symbolic expression (string of symbols) | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper focuses on symbolic regression: given tabular datasets of variable values and outputs, the system discovers a symbolic formula matching the data. Inputs are described as tables of rows of variables, supporting a 2D (x, y) input structure, while outputs are symbolic expressions represented as symbol strings (1D). The method constructs intermediate representations via neural-network-based symmetry/separability discovery, but the paper does not specify attention dynamics or explicit size caps for input tables; output length is searched over increasing string lengths.

## Evidence
### Task: symbolic regression (discover symbolic expression from data)
- "we are given a table of numbers, whose rows are of the form  $\{x_1, ..., x_n, y\}$" (Introduction)
- "our task is to discover the correct symbolic expression for the unknown mystery function f" (Introduction)
- "representing them as strings of symbols, trying first all strings of length 1, then all of length 2" (Section II.D Brute Force)
- Inference: In Dimension set to 2D because the input is a "table of numbers" with rows (Introduction); Out Dimension/Out Dynamics inferred from "strings of symbols" searched by length (Section II.D Brute Force); State Dynamic inferred from "using neural networks to discover hidden simplicity such as symmetry or separability in the mystery data" (Introduction).
