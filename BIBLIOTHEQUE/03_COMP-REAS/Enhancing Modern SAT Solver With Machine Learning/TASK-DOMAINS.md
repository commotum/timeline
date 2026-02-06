# Enhancing Modern SAT Solver With Machine Learning Method (2025)
Source: Enhancing Modern SAT Solver With Machine Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification (backbone variables) | Boolean formula encoded as WLIG graph | 2D (x, y) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | per-variable backbone probabilities | 1D (t) (inferred) | Open (inferred) |
| classification (UNSAT-core variables) | Boolean formula encoded as WLIG graph | 2D (x, y) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | per-variable UNSAT-core probabilities | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper uses GNNs to classify variables as backbone variables for SAT instances and as UNSAT-core variables for UNSAT instances, using CNF formulas encoded as WLIG graphs. Outputs are per-variable probability scores that guide solver decisions. Dimensions are inferred as 2D for adjacency-matrix graph inputs and 1D for per-variable outputs, with Open dynamics because formulas vary in size. Attention is treated as Static and state as Constructed (both inferred) based on full-graph message passing and learned node embeddings.

## Evidence
### Task: classification (backbone variables)
- "we present a GNN-based algorithm that predicts at the same time backbone variables for SAT instances and UNSAT-core variables for UNSAT instances." (Abstract)
- "The two GNN models designed for classification tasks are trained on distinct datasets" (Section 4.1 Overview)
- "An input Boolean formula is initially converted into a Weighted Literal-Incidence Graph (WLIG) and fed into a Graph Neural Network (GNN)" (Section 4.1 Overview)
- "The GNN generates probabilities indicating the likelihood of variables being part of the backbone or the UNSAT-core." (Section 4.1 Overview)
- Inference: In Dimension 2D (x, y) (inferred) because the model uses "the adjacency matrix as input" (Section 4.4); In/Out Dynamics Open (inferred) because variable counts range "from five to 41647" (Section 5.1); Out Dimension 1D (t) (inferred) because it estimates "the probability  p_v  of each variable v" (Section 4.4); Attention Static (inferred) and State Constructed (inferred) because "node embedding vectors H are updated by aggregating embeddings from their neighboring nodes" (Section 4.4)

### Task: classification (UNSAT-core variables)
- "we present a GNN-based algorithm that predicts at the same time backbone variables for SAT instances and UNSAT-core variables for UNSAT instances." (Abstract)
- "The two GNN models designed for classification tasks are trained on distinct datasets" (Section 4.1 Overview)
- "An input Boolean formula is initially converted into a Weighted Literal-Incidence Graph (WLIG) and fed into a Graph Neural Network (GNN)" (Section 4.1 Overview)
- "The GNN generates probabilities indicating the likelihood of variables being part of the backbone or the UNSAT-core." (Section 4.1 Overview)
- Inference: In Dimension 2D (x, y) (inferred) because the model uses "the adjacency matrix as input" (Section 4.4); In/Out Dynamics Open (inferred) because variable counts range "from five to 41647" (Section 5.1); Out Dimension 1D (t) (inferred) because it estimates "the probability  p_v  of each variable v" (Section 4.4); Attention Static (inferred) and State Constructed (inferred) because "node embedding vectors H are updated by aggregating embeddings from their neighboring nodes" (Section 4.4)
