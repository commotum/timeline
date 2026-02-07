# Training Verifiers to Solve Math Word Problems (Not specified in the paper)
Source: GSM8K- Training Verifiers to Solve Math Word Problems.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Generation (math word problem solutions) | Grade school math word problem text (natural language) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Natural language solution text with final answer | 1D (t) (inferred) | Capped (inferred) |
| Classification (solution correctness verification) | Problem text + candidate solution text | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Correctness probability / label | 0D (inferred) | Fixed (inferred) |

## Summary
The paper targets grade school math word problems, with models generating full natural language solutions and final answers from problem text. It also trains verifiers that take a problem plus a candidate solution and output a correctness probability for ranking. Attention and state dynamics are not specified, while generation uses a capped maximum sample length (400 tokens).

## Evidence
### Task: Generation (math word problem solutions)
- "a curated dataset of 8.5K grade school math questions and natural language solutions" (Section 1 Introduction)
- "At test time, we judge performance by autoregressively sampling a single low temperature solution and checking whether the final answer is correct." (Section 4 Methods)
- "We collect solutions in natural language rather than as pure math expressions." (Section 2 Dataset)
- "| Max Sample Length             | 400 tokens                             |  |" (Appendix B Hyperparameters)
- Inference: Inferred 1D (t) input/output because the problems and solutions are natural language text sequences; inferred capped output dynamics from the stated max sample length. (Section 1 Introduction; Section 2 Dataset; Appendix B Hyperparameters)

### Task: Classification (solution correctness verification)
- "We propose training verifiers to evaluate the correctness of model generated solutions." (Abstract)
- "Conditioned on the problem and a candidate solution, the verifier outputs the probability that the solution is correct." (Section 4.2 Verification)
- Inference: Inferred 1D (t) input because the problem and candidate solution are text sequences; inferred 0D fixed output because the verifier outputs a single probability. (Abstract; Section 4.2 Verification)

## CSV Output (required)
/home/jake/Developer/timeline/BIBLIOTHEQUE/03_COMP-REAS/GSM8K- Training Verifiers to Solve Math Word Problems/.TASK-DOMAINS.csv.tmp.d04770dc80c142db834dbb0d8e6e371f
