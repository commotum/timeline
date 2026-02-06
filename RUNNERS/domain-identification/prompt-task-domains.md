## Task Domains OCR Review Prompt

You are reviewing an OCR markdown extraction of a research paper.
Your goal is to extract and classify the task domains the model covers, using the
provided glossary definitions and supplemental outputs as hints.

Input:
- OCR markdown path: [SOURCE_MD_ABS_PATH]
- Glossary reference path: [REFERENCE_ABS_PATH]
- Supplemental output (Extending Dimensions): [EXTENDING_DIMENSIONS_MD_ABS_PATH]
- Supplemental output (Task-Model Ratio): [TASK_MODEL_RATIO_MD_ABS_PATH]

Output:
- Write your final markdown to: [MD_ABS_PATH]
- Write the task table as CSV to: [CSV_ABS_PATH]

Rules:
1. Read the entire OCR markdown. Use only this text and the glossary file as
   source of truth. Do not consult PDFs or CLASS_*.md files.
2. The supplemental outputs are a starting point only. Use them to find
   candidate tasks, but confirm everything against the OCR markdown. If the OCR
   does not support a claim, mark it as "Not specified in the paper." If a
   supplemental file is missing or empty, ignore it.
3. The glossary file defines Task, Input, Output, Dimension, Dynamics,
   Attention, and State. Use its definitions and labels verbatim.
4. Quote the paper verbatim and include the section name or page number when
   available.
5. If information is not explicitly stated, you may infer values only when the
   OCR text provides clear support (e.g., architectural description). Any
   inferred value must be marked with " (inferred)" in the table/CSV and noted
   in Evidence as an inference. If you cannot infer confidently, write
   "Not specified in the paper."
6. Do not count ablations, architectural variants, or hyperparameter sweeps as
   separate tasks unless they are distinct tasks by intent.
7. You must first build a complete list of every task the model handles with
   the required fields (Task, Input, In Dimension, In Dynamics, Attention
   Dynamic, State Dynamic, Output, Out Dimension, Out Dynamics). Only after
   that list is complete should you write the summary and evidence sections.
8. Follow the output format below exactly. Include every section.
9. Write the markdown to [MD_ABS_PATH] and the CSV to [CSV_ABS_PATH].
   Do not include extra sections or commentary.

---

## Output format (required)

# <Paper Title> (<Year>)
Source: <OCR markdown filename or identifier>

## Task Table
Use this exact header and column order:

| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

Guidance for values:
- Task: concise task intent (e.g., classification, detection, generation,
  prediction, control, manipulation).
- Input/Output: conceptual objects (e.g., tokens, images, tables, trajectories,
  sensor streams).
- In Dimension / Out Dimension: use the glossary labels (0D, 1D (t), 2D (x, y),
  3D (x, y, z) or (x, y, t), 4D (x, y, z, t)). Use semicolons if multiple apply.
- In Dynamics / Out Dynamics: Fixed, Capped, or Open (from glossary).
- Attention Dynamic: Static or Dynamic (from glossary).
- State Dynamic: Direct or Constructed (from glossary).

If the paper does not specify any tasks, include exactly one row with
"Not specified in the paper." in every column.

## Summary
Write 2-4 sentences that summarize the overall task/modality coverage and the
range of Dimension/Dynamics/Attention/State you can justify from the paper.

## Evidence
Provide evidence quotes for each task row. Use this structure:

### Task: <Task>
- "<Quote>" (p. X or Section ...)
- "<Quote>" (p. Y or Section ...)
- Inference: <What you inferred and why> (cite supporting text; only if any value is inferred)

---

## CSV Output (required)
Write a CSV file to [CSV_ABS_PATH] with the same rows and columns as the Task
Table. Use the exact header:

task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic

Do not add extra columns, commentary, or blank lines.
