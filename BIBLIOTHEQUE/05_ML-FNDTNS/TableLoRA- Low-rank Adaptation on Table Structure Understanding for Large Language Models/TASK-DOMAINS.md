# TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models (Not specified in the paper)
Source: TableLoRA- Low-rank Adaptation on Table Structure Understanding for Large Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Table question answering | Table and related query text | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer words/sentences (token sequence) | 1D (t) (inferred) | Capped (inferred) |
| Table fact verification | Table and related statement text | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Truthfulness judgment (bool) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates TableLoRA on two table-centric tasks: table question answering and table fact verification. Inputs combine 2D table structure with 1D text (queries or statements), while outputs span 1D generated answers and 0D boolean judgments. Based on the reported sequence and table-size limits, the task interfaces are capped rather than open. The described transformer/LoRA processing supports static attention and direct state use in this setting.

## Evidence
### Task: Table question answering
- "Tabular tasks involve generating an answer sequence output given a table T and related text text (such as questions, table captions, etc.)." (Section 3 Methodology)
- "The first three are Table QA datasets, where the input consists of a table and a related query, and the task is to answer the query based on the table, with the output being the answer to the question." (Section 4.1 Experiment Setup)
- Inference: `In Dimension = 2D (x, y); 1D (t)` and `Out Dimension = 1D (t)` are inferred from the explicit table-plus-text input and answer-sequence output. `In/Out Dynamics = Capped` is inferred from "maximum sequence length of 4,000 tokens ... and 1,000 tokens" (Section B.2) and "maximum values for columns and rows are set to 40 and 600 ... and 50 and 600" (Section B.3). `Attention Dynamic = Static` and `State Dynamic = Direct` are inferred from the fixed transformer/LoRA forward computation described in Section 3.2 (parallel LoRA pathways with per-layer embeddings, no runtime input-selection mechanism or persistent external state).

### Task: Table fact verification
- "The last dataset, TabFact, is for fact verification, where the input is a table and a related statement, and the task is to determine the truthfulness of the statement based on the table, with the output being the judgment result." (Section 4.1 Experiment Setup)
- "| TabFact | flat | fact/statement | bool |" (Section A.1 Datasets Selection, Table 4)
- Inference: `In Dimension = 2D (x, y); 1D (t)` is inferred from table + statement input. `Out Dimension = 0D` and `Out Dynamics = Fixed` are inferred from boolean judgment output. `In Dynamics = Capped` is inferred from the same explicit maxima in Section B.2 and Section B.3. `Attention Dynamic = Static` and `State Dynamic = Direct` are inferred from the same model-design evidence in Section 3.2.
