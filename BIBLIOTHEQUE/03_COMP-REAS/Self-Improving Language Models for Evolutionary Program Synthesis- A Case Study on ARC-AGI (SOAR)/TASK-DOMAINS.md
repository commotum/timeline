# Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI (2025)
Source: Self-Improving Language Models for Evolutionary Program Synthesis- A Case Study on ARC-AGI (SOAR).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Generation (Python program synthesis for ARC) | Colored ARC training input-output grid pairs and test input grids | 2D (x, y) | Capped | Static (inferred) | Direct (inferred) | Python transformation programs | 1D (t) (inferred) | Not specified in the paper. |
| Manipulation (Python program refinement/code repair) | Candidate Python program `f`, ARC grids (`x_train`, `y_train`, `x_test`), and synthesized outputs `y_synth` | 1D (t) (inferred); 2D (x, y) | Capped | Dynamic | Constructed (inferred) | Refined Python programs `f^+` | 1D (t) (inferred) | Not specified in the paper. |
| Prediction (ARC test output-grid selection via weighted voting) | Program-produced test output grids and training-example accuracies from candidate solutions | 2D (x, y); 0D | Capped | Dynamic | Constructed | Two selected candidate test output grids | 2D (x, y) | Capped |

## Summary
The paper covers ARC as an inductive program-synthesis setting over colored 2D grids, where the system generates Python transformation programs and then predicts test output grids. Modalities therefore span 2D grid inputs, 1D code outputs, and 2D grid outputs. Dynamics are capped for inputs by explicit ARC limits (2-10 examples, grid sizes 1..30) and capped for final predictions by fixed search/selection budgets, while control behavior ranges from static/direct sampling calls to dynamic/constructed refinement and ensembling.

## Evidence
### Task: Generation (Python program synthesis for ARC)
- "- 1. a set of 2–10 training examples  $\{(x_{\text{train}}, y_{\text{train}})\}$  where  $x_{\text{train}}$  and  $y_{\text{train}}$  are colored grid pairs;" (Section 3.1)
- "The goal is to find a Python function f such that  $f(x_{\text{train}}) = y_{\text{train}}$  for all training examples, and  $f(x_{\text{test}})$  produces the correct (hidden)  $y_{\text{test}}$ ." (Section 3.1)
- "**Program sampling.** Given a base LLM parameterized by  $\theta$ , we sample a set of Python programs f without constraining ourselves to a hand-coded domain-specific language:" (Section 3.2)
- Inference: Attention Dynamic is marked Static (inferred) and State Dynamic as Direct (inferred) because sampling is defined as one conditional generation, "$f \sim P_{\theta}(\cdot \mid x_{\text{test}}, x_{\text{train}}, y_{\text{train}}).$" with no runtime selection/retrieval mechanism described in this step (Section 3.2). Out Dimension is marked 1D (t) (inferred) because the generated object is a Python program.

### Task: Manipulation (Python program refinement/code repair)
- "**Program refinement.** When a candidate program f produces incorrect outputs  $(y_{\text{synth}} = f(x_{\text{train}}) \neq y_{\text{train}})$ , we can use this execution feedback to guide the LLM in refining its solution  $f \to f^+$ :" (Section 3.2)
- "$$f^+ \sim P_{\theta}(\cdot \mid f, x_{\text{test}}, x_{\text{train}}, y_{\text{train}}, y_{\text{synth}}),$$" (Section 3.2)
- "The second step frames refinement as a generative multi-armed bandit: each refinement creates a new arm that can further be refined." (Section 3.2)
- Inference: In Dimension includes 1D (t) (inferred) because refinement explicitly conditions on a candidate program `f` and returns `f^+` code; State Dynamic is marked Constructed (inferred) because refinement maintains and extends an iterative arm structure during search (Section 3.2).

### Task: Prediction (ARC test output-grid selection via weighted voting)
- "In the end, we use majority voting to select the most likely test output grids to submit for evaluation (see Appendix D.1)." (Section 3.2)
- "Each program is evaluated on the ARC task's input-output examples to compute its example accuracy, and is also run on the test input to produce an output grid." (Section 3.2)
- "The algorithm processes the ensemble of model responses, each containing output predictions for a set of input grids. It groups responses by their test output grids and applies weighted voting to select the most reliable predictions (see Alg. 1)." (Section D.1)
