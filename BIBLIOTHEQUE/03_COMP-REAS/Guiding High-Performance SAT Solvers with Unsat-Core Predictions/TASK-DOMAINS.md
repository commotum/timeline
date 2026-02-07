# Guiding High-Performance SAT Solvers with Unsat-Core Predictions (Not specified in the paper.)
Source: Guiding High-Performance SAT Solvers with Unsat-Core Predictions.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unsat-core variable prediction | CNF clauses/literals (Boolean formula) | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | variable scores / core-membership probabilities | 1D (t) (inferred) | Capped (inferred) |
| cubing decision prediction | SAT problems for cubing (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | cubing decisions / cubes (branch literals) (inferred) | Not specified in the paper. | Not specified in the paper. |
| variable phase prediction | satisfiable SAT problems | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | variable phase predictions | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper's primary model task is predicting unsat-core variable membership from CNF SAT instances, producing per-variable scores/probabilities. The input is represented as a clause-literal matrix and the output is a variable-length score vector, supporting a 2D-to-1D mapping with capped dynamics, static attention, and constructed internal state (inferred). The discussion also mentions exploratory tasks of learning cubing decisions and predicting variable phases for satisfiable models, but their structural dimensions and dynamics are not specified.

## Evidence
### Task: unsat-core variable prediction
- "we train a simplified NeuroSAT architecture to directly predict the unsatisfiable cores of real problems." (Abstract)
- "generate a supervised dataset mapping unsatisfiable problems to the variables in their unsatisfiable cores." (Section 1 Introduction)
- "The vector  $\hat{v}$  is the output of NeuroCore, and consists of a numerical score for each variable" (Section 3 Neural Network Architecture)
- "We represent a Boolean formula in CNF with  $n_v$  variables and  $n_c$ clauses by an  $n_c \times 2n_v$  sparse matrix  $\mathcal{G}$" (Section 3 Neural Network Architecture)
- "exceed a fixed cutoff (we used 10 million)." (Section 4 Hybrid Solving)
- "periodically query NeuroSAT on the *entire* problem (i.e. not conditioning on the current trail)" (Section 1 Introduction)
- "the network performs T iterations of \"message passing\"" (Section 3 Neural Network Architecture)
- Inference: Labeled input/output dimensions as 2D/1D and dynamics as Capped because the formula is represented as an $n_c \times 2n_v$ matrix and queries are limited by a fixed cutoff; attention is Static because they query the entire problem; state is Constructed due to iterative message passing.

### Task: cubing decision prediction
- "We also experimented with training NeuroSAT to imitate the decisions of the March cubing heuristic." (Section 7 Discussion)
- "the only competitive heuristic we were able to learn *de novo* was a cubing strategy for uniform random problems." (Section 7 Discussion)
- Inference: Labeled input as SAT problems and output as cubing decisions/cubes because they describe imitating the March cubing heuristic and learning a cubing strategy.

### Task: variable phase prediction
- "we used existing solvers to find models of satisfiable problems, and then trained NeuroSAT to predict the phases of each of the variables individually." (Section 7 Discussion)
- "instrumented MiniSat to choose the phase of each decision variable in proportion to NeuroSAT's prediction." (Section 7 Discussion)

## CSV Output (required)
task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic
unsat-core variable prediction,CNF clauses/literals (Boolean formula),"2D (x, y) (inferred)",Capped (inferred),Static (inferred),Constructed (inferred),variable scores / core-membership probabilities,1D (t) (inferred),Capped (inferred)
cubing decision prediction,SAT problems for cubing (inferred),Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,cubing decisions / cubes (branch literals) (inferred),Not specified in the paper.,Not specified in the paper.
variable phase prediction,satisfiable SAT problems,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,Not specified in the paper.,variable phase predictions,Not specified in the paper.,Not specified in the paper.
