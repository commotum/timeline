# NEURAL PROGRAMMER: INDUCING LATENT PROGRAMS WITH GRADIENT DESCENT (Not specified in the paper.)
Source: Neural Programmer- Inducing Latent Programs with Gradient Descent.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Table question answering (scalar answer) | Question tokens; table | 1D (t) (inferred); 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Scalar value | 0D (inferred) | Fixed (inferred) |
| Table question answering (table lookup) | Question tokens; table | 1D (t) (inferred); 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | List of table items (lookup) | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
The paper applies Neural Programmer to question answering on tables, using question-table-answer triples from a synthetic table-comprehension dataset. It supports both scalar numeric answers (e.g., sums/differences) and table lookup outputs that select items from the table. Inputs combine a 1D question sequence and a 2D table with variable rows/columns, while outputs are 0D scalars or 2D table selections (inferred). The model selects operations/columns step-by-step and maintains a history/memory state, implying dynamic attention and constructed state (inferred).

## Evidence
### Task: Table question answering (scalar answer)
- "we apply Neural Programmer to the task of question answering on tables" (Section 2)
- "training set consisting of triples, where each triple has a question, a data source and an answer." (Section 2)
- "data source is in the form of a table,  $table \in \mathbb{R}^{M \times C}$" (Section 2)
- "Neural Programmer currently supports two types of outputs: a) a scalar output" (Section 2.3)
- "The first type of output is for questions of type \"Sum of elements in column C\"" (Section 2.3)
- Inference: In Dimension from "question, a data source and an answer" and "table \in \mathbb{R}^{M \times C}" (Section 2); In Dynamics Capped from "M and C can vary amongst examples" (Section 2), "The number of rows is sampled randomly from [30, 100] in training" and "max_columns = 3, 5 or 10" (Section 3.1); Attention Dynamic from "At each step, it can select a segment in the data source" (Introduction); State Dynamic from "The history RNN keeps track of the previous operations and columns selected" (Section 2.4); Out Dimension/Out Dynamics from "scalar_answer_t \in \mathbb{R}" (Section 2.3).

### Task: Table question answering (table lookup)
- "we apply Neural Programmer to the task of question answering on tables" (Section 2)
- "training set consisting of triples, where each triple has a question, a data source and an answer." (Section 2)
- "Neural Programmer currently supports two types of outputs: a) a scalar output, and b) a list of items selected from the table" (Section 2.3)
- "The second type of output is for questions of type \"Print elements in column A that are greater than 50.\"" (Section 2.3)
- "lookup_answer_t \in [0,1]^{M \times C}" (Section 2.3)
- Inference: In Dimension from "question, a data source and an answer" and "table \in \mathbb{R}^{M \times C}" (Section 2); In Dynamics Capped from "M and C can vary amongst examples" (Section 2), "The number of rows is sampled randomly from [30, 100] in training" and "max_columns = 3, 5 or 10" (Section 3.1); Attention Dynamic from "At each step, it can select a segment in the data source" (Introduction); State Dynamic from "The history RNN keeps track of the previous operations and columns selected" (Section 2.4); Out Dimension/Out Dynamics from "lookup_answer_t \in [0,1]^{M \times C}" (Section 2.3).

---

## CSV Output (required)
