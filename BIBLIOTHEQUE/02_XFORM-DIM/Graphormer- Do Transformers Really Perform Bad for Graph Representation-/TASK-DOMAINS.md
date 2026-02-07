# Do Transformers Really Perform Bad for Graph Representation? (Not specified in the paper.)
Source: Graphormer- Do Transformers Really Perform Bad for Graph Representation-.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Regression (graph-level property prediction; PCQM4M-LSC) | 2D molecular graphs | Not specified in the paper. | Not specified in the paper. | Static (inferred) | Direct (inferred) | HOMO-LUMO energy gap | 0D (inferred) | Fixed (inferred) |
| Binary classification (molecular property prediction; OGBG-MolPCBA) | molecular graphs | Not specified in the paper. | Not specified in the paper. | Static (inferred) | Direct (inferred) | binary property label | 0D (inferred) | Fixed (inferred) |
| Binary classification (molecular property prediction; OGBG-MolHIV) | molecular graphs | Not specified in the paper. | Not specified in the paper. | Static (inferred) | Direct (inferred) | binary property label | 0D (inferred) | Fixed (inferred) |
| Regression (graph property prediction for constrained solubility; ZINC) | molecular graphs | Not specified in the paper. | Not specified in the paper. | Static (inferred) | Direct (inferred) | constrained solubility | 0D (inferred) | Fixed (inferred) |
| Node representation extraction (future work) | graph structured data | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | node representations (inferred) | Not specified in the paper. | Not specified in the paper. |

## Summary
Graphormer is evaluated on graph-level molecular property prediction tasks spanning regression and binary classification across PCQM4M-LSC, OGBG-MolPCBA, OGBG-MolHIV, and ZINC. Inputs are molecular graphs and outputs are scalar properties or binary labels. The paper also notes node representation extraction on graph structured data as future work without specifying dimensional or dynamic constraints. The paper does not explicitly specify input dimensional indexing or interface dynamics for the evaluated tasks; attention and state are inferred from the described Transformer self-attention architecture.

## Evidence
### Task: Regression (graph-level property prediction; PCQM4M-LSC)
- "PCQM4m-LSC is a quantum chemistry graph-level prediction task" (Section B.1 Details of Datasets)
- "predict DFT(density functional theory)-calculated HOMO-LUMO energy gap of molecules given their 2D molecular graphs" (Section B.1 Details of Datasets)
- "PCQM4M-LSC     | Large  | 3,803,453 | 53,814,542 | 55,399,880 | Regression" (Table 6, Section B.1 Details of Datasets)
- Inference: Attention Dynamic marked Static (inferred) because "each node can attend to all other nodes in the graph"; State Dynamic marked Direct (inferred) because "Graphormer is built upon the original implementation of classic Transformer encoder"; Out Dimension marked 0D (inferred) and Out Dynamics Fixed (inferred) because the task predicts a single energy-gap value per graph. (Section 3.1.2 Spatial Encoding; Section 3.2 Implementation Details of Graphormer; Section B.1 Details of Datasets)

### Task: Binary classification (molecular property prediction; OGBG-MolPCBA)
- "two molecular graph datasets in popular OGB leaderboards, i.e., OGBG-MolPCBA and OGBG-MolHIV." (Section B.1 Details of Datasets)
- "OGBG-MolPCBA   | Medium | 437,929   | 11,386,154 | 12,305,805 | Binary classification" (Table 6, Section B.1 Details of Datasets)
- Inference: Attention Dynamic marked Static (inferred) because "each node can attend to all other nodes in the graph"; State Dynamic marked Direct (inferred) because "Graphormer is built upon the original implementation of classic Transformer encoder"; Out Dimension marked 0D (inferred) and Out Dynamics Fixed (inferred) because the task is "Binary classification" (single label per graph). (Section 3.1.2 Spatial Encoding; Section 3.2 Implementation Details of Graphormer; Table 6, Section B.1 Details of Datasets)

### Task: Binary classification (molecular property prediction; OGBG-MolHIV)
- "two molecular graph datasets in popular OGB leaderboards, i.e., OGBG-MolPCBA and OGBG-MolHIV." (Section B.1 Details of Datasets)
- "OGBG-MolHIV    | Small  | 41,127    | 1,048,738  | 1,130,993  | Binary classification" (Table 6, Section B.1 Details of Datasets)
- Inference: Attention Dynamic marked Static (inferred) because "each node can attend to all other nodes in the graph"; State Dynamic marked Direct (inferred) because "Graphormer is built upon the original implementation of classic Transformer encoder"; Out Dimension marked 0D (inferred) and Out Dynamics Fixed (inferred) because the task is "Binary classification" (single label per graph). (Section 3.1.2 Spatial Encoding; Section 3.2 Implementation Details of Graphormer; Table 6, Section B.1 Details of Datasets)

### Task: Regression (graph property prediction for constrained solubility; ZINC)
- "We use the ZINC datasets, which is the most popular real-world molecular dataset to predict graph property regression for contrained solubility" (Section B.1 Details of Datasets)
- "ZINC (sub-set) | Small  | 12,000    | 277,920    | 597,960    | Regression" (Table 6, Section B.1 Details of Datasets)
- Inference: Attention Dynamic marked Static (inferred) because "each node can attend to all other nodes in the graph"; State Dynamic marked Direct (inferred) because "Graphormer is built upon the original implementation of classic Transformer encoder"; Out Dimension marked 0D (inferred) and Out Dynamics Fixed (inferred) because the task predicts a single solubility value per graph. (Section 3.1.2 Spatial Encoding; Section 3.2 Implementation Details of Graphormer; Section B.1 Details of Datasets)

### Task: Node representation extraction (future work)
- "There is a wide range of node representation tasks on graph structured data, such as finance, social network, and temporal prediction." (Section D Discussion & Future Work)
- "Graphormer could be naturally used for node representation extraction with an applicable graph sampling strategy." (Section D Discussion & Future Work)
- Inference: Output marked node representations (inferred) because the task is described as "node representation extraction." (Section D Discussion & Future Work)
