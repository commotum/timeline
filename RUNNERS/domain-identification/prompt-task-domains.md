## Task Domains OCR Review Prompt

You are reviewing an OCR markdown extraction of a research paper.
Your goal is to extract and classify the task domains the model covers, using the
provided glossary definitions.

Input:
- OCR markdown path: [SOURCE_MD_ABS_PATH]
- Glossary reference path: [REFERENCE_ABS_PATH]

Output:
- Write your final markdown to: [MD_ABS_PATH]
- Write the task table as CSV to: [CSV_ABS_PATH]

Rules:
1. Read the entire OCR markdown. Use only this text and the glossary file.
   Do not consult PDFs or CLASS_*.md files.
2. The glossary file defines Task, Input, Output, Dimension, Dynamics,
   Attention, and State. Use its definitions and labels verbatim.
3. Quote the paper verbatim and include the section name or page number when
   available.
4. If information is not explicitly stated, write "Not specified in the paper."
5. Do not count ablations, architectural variants, or hyperparameter sweeps as
   separate tasks unless they are distinct tasks by intent.
6. Follow the output format below exactly. Include every section.
7. Write the markdown to [MD_ABS_PATH] and the CSV to [CSV_ABS_PATH].
   Do not include extra sections or commentary.

---

## Output format (required)

# <Paper Title> (<Year>)
Source: <OCR markdown filename or identifier>

## Summary
Write 2-4 sentences that summarize the overall task/modality coverage and the
range of Dimension/Dynamics/Attention/State you can justify from the paper.

## Task Table
Use this exact header and column order:

| Task | Input | Output | Dimension | Dynamics | Attention | State |
| --- | --- | --- | --- | --- | --- | --- |
| ... | ... | ... | ... | ... | ... | ... |

Guidance for values:
- Task: concise task intent (e.g., classification, detection, generation,
  prediction, control, manipulation).
- Input/Output: conceptual objects (e.g., tokens, images, tables, trajectories,
  sensor streams).
- Dimension: use the glossary labels (0D, 1D (t), 2D (x, y), 3D (x, y, z) or
  (x, y, t), 4D (x, y, z, t)). Use semicolons if multiple apply.
- Dynamics: Fixed, Capped, or Open (from glossary).
- Attention: Static or Dynamic (from glossary).
- State: Direct or Constructed (from glossary).

If the paper does not specify any tasks, include exactly one row with
"Not specified in the paper." in every column.

## Evidence
Provide evidence quotes for each task row. Use this structure:

### Task: <Task>
- "<Quote>" (p. X or Section ...)
- "<Quote>" (p. Y or Section ...)

---

## CSV Output (required)
Write a CSV file to [CSV_ABS_PATH] with the same rows and columns as the Task
Table. Use the exact header:

Task,Input,Output,Dimension,Dynamics,Attention,State

Do not add extra columns, commentary, or blank lines.
