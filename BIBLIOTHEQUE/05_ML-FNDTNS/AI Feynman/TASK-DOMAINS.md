# AI Feynman: A physics-inspired method for symbolic regression (2020)
Source: AI Feynman.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| symbolic regression (discover analytic expression for f) | data table of numeric tuples {x1,...,xn,y}; optionally unit table of physical units | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | analytic/symbolic expression for f | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper addresses symbolic regression: discovering analytic expressions from tabular numeric data, optionally with unit tables. Inputs are described as tables of rows of tuples and outputs as symbolic expressions, supporting inferred 2D inputs and 1D symbolic outputs, while attention dynamics are not specified. The method constructs internal models via neural network interpolation, so state is inferred as constructed.

## Evidence
### Task: symbolic regression (discover analytic expression for f)
- "we are given a table of numbers, whose rows are of the form  $\{x_1,..., x_n, y\}$" (INTRODUCTION)
- "our task is to discover the correct symbolic expression for the unknown mystery function f, optionally including the complication of noise." (INTRODUCTION)
- "its task is to predict *f* for each mystery taking the data table (and optionally the unit table) as input." (The Feynman Symbolic Regression Database)
- "the challenge is to discover the correct analytic expression for the mystery function f." (The Feynman Symbolic Regression Database)
- Inference: In Dimension is 2D (x, y) because the input is a "table of numbers" with "rows" of tuples; Out Dimension is 1D (t) because expressions are represented as "strings of symbols"; State Dynamic is Constructed because "we train a neural network to predict the output given its input." (INTRODUCTION; Brute force; Neural network training)
