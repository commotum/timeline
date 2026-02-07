# LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS (Not specified in the paper.)
Source: LoRA- Low-Rank Adaptation of Large Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Summarization (articles) | content of an article | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | summary | 1D (t) (inferred) | Not specified in the paper. |
| Machine reading comprehension (MRC) | sequences of tokens | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | sequences of tokens | 1D (t) (inferred) | Not specified in the paper. |
| Natural language to SQL generation (NL2SQL/WikiSQL) | natural language question/query; table schema | 1D (t); 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | SQL command/query | 1D (t) (inferred) | Not specified in the paper. |
| Natural language inference (MNLI/QNLI/RTE) | natural language text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | inference label (inferred) | 0D (inferred) | Not specified in the paper. |
| Sentiment analysis (SST-2) | natural language text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | sentiment label (inferred) | 0D (inferred) | Not specified in the paper. |
| Paraphrase detection (MRPC) | text pair (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | paraphrase label (inferred) | 0D (inferred) | Not specified in the paper. |
| Linguistic acceptability (CoLA) | natural language text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | acceptability label (inferred) | 0D (inferred) | Not specified in the paper. |
| Question-answering (QQP) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Textual similarity (STS-B) | text pair (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | similarity score (inferred) | 0D (inferred) | Not specified in the paper. |
| Conversation summarization (SAMSum) | staged chat conversations | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | abstractive summaries | 1D (t) (inferred) | Not specified in the paper. |
| Data-to-text generation (E2E NLG Challenge) | sequence of slot-value pairs | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | natural language reference text | 1D (t) (inferred) | Not specified in the paper. |
| Data-to-text generation (DART) | sequence of ENTITY — RELATION — ENTITY triples | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | natural language text (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Data-to-text generation (WebNLG) | sequence of SUBJECT — PROPERTY — OBJECT triples | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | natural language text (inferred) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper frames LoRA for conditional text generation tasks (summarization, MRC, NL2SQL/SQL generation) and evaluates it on GLUE NLU tasks plus data-to-text generation (E2E, DART, WebNLG) and conversation summarization (SAMSum). Inputs are primarily natural language sequences or structured slot-value/triple sequences, with SQL outputs for NL2SQL/WikiSQL and summaries for summarization tasks; dimensions are therefore mostly 1D sequences, with tabular schema elements for WikiSQL. The paper does not specify task-level dynamics, attention dynamics, or state dynamics, so those are marked as not specified.

## Evidence
### Task: Summarization (articles)
- "Consider adapting this pre-trained model to downstream conditional text generation tasks, such as summarization, machine reading comprehension (MRC), and natural language to SQL (NL2SQL)." (Section 2 Problem Statement)
- "for summarization,  $x_i$  is the content of an article and  $y_i$  its summary." (Section 2 Problem Statement)
- "where both  $x_i$  and  $y_i$  are sequences of tokens." (Section 2 Problem Statement)
- Inference: Mapped token sequences to 1D (t) input/output dimensions based on the paper stating x_i and y_i are sequences of tokens. (Section 2 Problem Statement)

### Task: Machine reading comprehension (MRC)
- "Consider adapting this pre-trained model to downstream conditional text generation tasks, such as summarization, machine reading comprehension (MRC), and natural language to SQL (NL2SQL)." (Section 2 Problem Statement)
- "where both  $x_i$  and  $y_i$  are sequences of tokens." (Section 2 Problem Statement)
- Inference: Treated MRC inputs/outputs as 1D (t) sequences because the paper states x_i and y_i are sequences of tokens. (Section 2 Problem Statement)

### Task: Natural language to SQL generation (NL2SQL/WikiSQL)
- "in NL2SQL,  $x_i$  is a natural language query and  $y_i$  its corresponding SQL command" (Section 2 Problem Statement)
- "The task is to generate SQL queries from natural language questions and table schemata." (Section C Dataset Details)
- "We encode context as  $x = \{\text{table schema, query}\}$  and target as  $y = \{\text{SQL}\}$ ." (Section C Dataset Details)
- Inference: Mapped natural language query/SQL command to 1D (t) and table schema to 2D (x, y) dimensions. (Section 2 Problem Statement; Section C Dataset Details)

### Task: Natural language inference (MNLI/QNLI/RTE)
- "GLUE Benchmark is a wide-ranging collection of natural language understanding tasks." (Section C Dataset Details)
- "MNLI (inference, Williams et al. (2018))" (Section C Dataset Details)
- "RTE (inference)" (Section C Dataset Details)
- Inference: Treated inference tasks as natural language text inputs with label outputs (0D) and mapped inputs to 1D (t). (Section C Dataset Details)

### Task: Sentiment analysis (SST-2)
- "GLUE Benchmark is a wide-ranging collection of natural language understanding tasks." (Section C Dataset Details)
- "SST-2 (sentiment analysis, Socher et al. (2013))" (Section C Dataset Details)
- Inference: Treated sentiment analysis as natural language text input with a label output (0D) and mapped inputs to 1D (t). (Section C Dataset Details)

### Task: Paraphrase detection (MRPC)
- "GLUE Benchmark is a wide-ranging collection of natural language understanding tasks." (Section C Dataset Details)
- "MRPC (paraphrase detection, Dolan & Brockett (2005))" (Section C Dataset Details)
- Inference: Treated paraphrase detection as text-pair input with a label output (0D) and mapped inputs to 1D (t). (Section C Dataset Details)

### Task: Linguistic acceptability (CoLA)
- "GLUE Benchmark is a wide-ranging collection of natural language understanding tasks." (Section C Dataset Details)
- "CoLA (linguistic acceptability, Warstadt et al. (2018))" (Section C Dataset Details)
- Inference: Treated linguistic acceptability as natural language text input with a label output (0D) and mapped inputs to 1D (t). (Section C Dataset Details)

### Task: Question-answering (QQP)
- "QQP<sup>8</sup> (question-answering)" (Section C Dataset Details)

### Task: Textual similarity (STS-B)
- "GLUE Benchmark is a wide-ranging collection of natural language understanding tasks." (Section C Dataset Details)
- "STS-B (textual similarity, Cer et al. (2017))." (Section C Dataset Details)
- Inference: Treated textual similarity as text-pair input with a similarity score output (0D) and mapped inputs to 1D (t). (Section C Dataset Details)

### Task: Conversation summarization (SAMSum)
- "It consists of staged chat conversations between two people and corresponding abstractive summaries written by linguists." (Section C Dataset Details)
- "target as  $y = \{\text{summary}\}$ ." (Section C Dataset Details)
- Inference: Mapped chat conversations and summaries to 1D (t) sequence dimensions. (Section C Dataset Details)

### Task: Data-to-text generation (E2E NLG Challenge)
- "Each sample input (x,y) consists of a sequence of slot-value pairs, along with a corresponding natural language reference text." (Section C Dataset Details)
- Inference: Mapped slot-value pairs and reference text to 1D (t) sequence dimensions. (Section C Dataset Details)

### Task: Data-to-text generation (DART)
- "DART is an open-domain data-to-text dataset described in Nan et al. (2020)." (Section C Dataset Details)
- "DART inputs are structured as sequences of ENTITY — RELATION — ENTITY triples." (Section C Dataset Details)
- Inference: Treated data-to-text as producing natural language text and mapped input/output to 1D (t) dimensions. (Section C Dataset Details)

### Task: Data-to-text generation (WebNLG)
- "WebNLG is another commonly used dataset for data-to-text evaluation (Gardent et al., 2017)." (Section C Dataset Details)
- "Each input example is represented by a sequence of SUBJECT — PROPERTY — OBJECT triples." (Section C Dataset Details)
- Inference: Treated data-to-text as producing natural language text and mapped input/output to 1D (t) dimensions. (Section C Dataset Details)
