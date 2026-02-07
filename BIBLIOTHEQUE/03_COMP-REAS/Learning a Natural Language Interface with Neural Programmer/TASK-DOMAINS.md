# Learning a Natural Language Interface with Neural Programmer (Not specified in the paper.)
Source: Learning a Natural Language Interface with Neural Programmer.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Question answering over database tables | natural language questions; database tables | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | numeric answers; table entries (cells) | 0D; 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper addresses natural language question answering over database tables using Neural Programmer. Inputs combine a token sequence question with a 2D table, while outputs are either scalar numeric answers or table-cell selections, implying mixed 0D/2D outputs (inferred). The model uses dynamic attention over the question and maintains constructed state across timesteps, and the table/question sizes vary but are experimentally bounded (inferred).

## Evidence
### Task: Question answering over database tables
- "apply it on WikiTableQuestions, a natural language question-answering dataset." (Abstract)
- "As input, a model receives a question along with a table (Figure 1)." (Section 1 Background and Introduction)
- "The variable lookup answer stores answers that are selected from the table while scalar answer stores numeric answers that are not provided in the table." (Section 2.2 Output and Row Selector)
- "Given an input table  $\Pi$ , containing M rows and C columns (M and C can vary across examples)" (Section 2.2 Output and Row Selector)
- "attention vector obtained by performing soft attention (Bahdanau et al., 2014) on the question using the history vector." (Section 2 Neural Programmer)
- "row selector, scalar answer and lookup answer which are updated at every timestep." (Section 2.2 Output and Row Selector)
- "We train only on examples for which the provided table has less than 100 rows" (Section 3.1 Data)
- Inference: In/Out Dimension and Dynamics are inferred from the question+table input, MxC table structure, lookup-answer indexing over (i,j), and the explicit training cap on rows; Attention Dynamic and State Dynamic are inferred from soft attention over the question and the updated row selector/answer variables (Sections 1, 2, 2.2, 3.1).
