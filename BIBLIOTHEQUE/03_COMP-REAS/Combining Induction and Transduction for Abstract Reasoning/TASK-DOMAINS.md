# Combining Induction and Transduction for Abstract Reasoning (Not specified in the paper)
Source: Combining Induction and Transduction for Abstract Reasoning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| few-shot grid transformation (output prediction) | training input-output grid pairs + test input grid (colored grids) | 2D (x, y) | Capped | Not specified in the paper. | Direct (inferred) | output grid (colored grid) | 2D (x, y) | Capped |
| program synthesis (function induction) | training input-output grid pairs + test input grid (colored grids) | 2D (x, y) | Capped | Not specified in the paper. | Constructed (inferred) | Python function/program f | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper targets ARC few-shot reasoning where models use a small set of input-output grid examples plus a test input grid to predict the output grid. Inputs/outputs are 2D colored grids with sizes between 1-30 pixels per side, so grid dimensions are 2D with capped dynamics. Induction is framed as program synthesis that outputs Python code (treated as 1D text, inferred), while attention dynamics are not specified and state dynamics are inferred from induction vs transduction descriptions.

## Evidence
### Task: few-shot grid transformation (output prediction)
- "Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input." (B.1 PROMPTING THE MODELS, Transduction example)
- "Every input from  $\mathcal{X}$  and output from  $\mathcal{Y}$  is a 2D grid ranging from 1–30 pixels per side." (Instantiating the framework for ARC)
- "Transduction directly predicts the test output, for example using a neural network." (Figure 2)
- Inference: State Dynamic marked Direct because the paper says transduction "directly predicts the test output" without constructing an intermediate function.

### Task: program synthesis (function induction)
- "Induction means first finding a function f where  $f(x_{\text{train}}) \approx y_{\text{train}}$ , and then predicting  $y_{\text{test}} = f(x_{\text{test}})$ ." (Introduction)
- "We represent functions f as Python code, meaning that induction synthesizes programs." (Introduction)
- "Therefore the induction model must generate Python code" (Instantiating the framework for ARC)
- Inference: State Dynamic marked Constructed because induction "first finding a function f" implies an explicit intermediate function; Out Dimension marked 1D (t) because the output is Python code (a textual sequence).
