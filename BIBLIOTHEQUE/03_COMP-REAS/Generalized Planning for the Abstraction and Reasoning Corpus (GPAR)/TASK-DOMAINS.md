# Generalized Planning for the Abstraction and Reasoning Corpus (2024)
Source: Generalized Planning for the Abstraction and Reasoning Corpus (GPAR).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Recoloring (object color change) | 2D grid images (training input-output pairs + test input grid) | 2D (x, y) | Capped (inferred) | Dynamic (inferred) | Constructed | 2D grid image (predicted output grid) | 2D (x, y) | Capped (inferred) |
| Movement (object position change) | 2D grid images (training input-output pairs + test input grid) | 2D (x, y) | Capped (inferred) | Dynamic (inferred) | Constructed | 2D grid image (predicted output grid) | 2D (x, y) | Capped (inferred) |
| Augmentation (object size/pattern change) | 2D grid images (training input-output pairs + test input grid) | 2D (x, y) | Capped (inferred) | Dynamic (inferred) | Constructed | 2D grid image (predicted output grid) | 2D (x, y) | Capped (inferred) |

## Summary
The paper addresses ARC object-centric abstract visual reasoning tasks that require learning from a few input-output grid pairs to generate the test output grid. The evaluated task types include recoloring, movement, and augmentation transformations over 2D pixel grids. The solver builds object-centric graph abstractions (constructed state) and executes planning programs with conditional branching (dynamic attention inferred); the DSL enumerates pixel coordinates up to 30x30, implying capped spatial dynamics (inferred).

## Evidence
### Task: Recoloring (object color change)
- "*recoloring* tasks which involve changing object colors;" (Experiments)
- "each task consists of a small set (typically three) of input-output image pairs for training" (Introduction)
- "Each image is a 2D grid of pixels with 10 possible colors." (Introduction)
- "The goal of the solver is to learn from the training instances how to generate the output for the test instance." (Figure 1 caption)
- "we represent an image as a graph of *nodes* representing objects and their spatial relations." (Abstraction over ARC)
- "pixel-0-0,, pixel-29-29." (Table 1)
- "conditional statements, and looping and branching structures allow the compact representation of solutions." (Generalized Planning)
- Inference: In/Out Dynamics = Capped (inferred) because pixel objects are enumerated from pixel-0-0 to pixel-29-29; Attention Dynamic = Dynamic (inferred) because planning programs use conditional, looping, and branching control.

### Task: Movement (object position change)
- "*movement* tasks which involve changing object positions;" (Experiments)
- "each task consists of a small set (typically three) of input-output image pairs for training" (Introduction)
- "Each image is a 2D grid of pixels with 10 possible colors." (Introduction)
- "The goal of the solver is to learn from the training instances how to generate the output for the test instance." (Figure 1 caption)
- "we represent an image as a graph of *nodes* representing objects and their spatial relations." (Abstraction over ARC)
- "pixel-0-0,, pixel-29-29." (Table 1)
- "conditional statements, and looping and branching structures allow the compact representation of solutions." (Generalized Planning)
- Inference: In/Out Dynamics = Capped (inferred) because pixel objects are enumerated from pixel-0-0 to pixel-29-29; Attention Dynamic = Dynamic (inferred) because planning programs use conditional, looping, and branching control.

### Task: Augmentation (object size/pattern change)
- "*augmentation*" (Experiments)
- "tasks which involve changing aspects of objects like size or pattern." (Experiments)
- "each task consists of a small set (typically three) of input-output image pairs for training" (Introduction)
- "Each image is a 2D grid of pixels with 10 possible colors." (Introduction)
- "The goal of the solver is to learn from the training instances how to generate the output for the test instance." (Figure 1 caption)
- "we represent an image as a graph of *nodes* representing objects and their spatial relations." (Abstraction over ARC)
- "pixel-0-0,, pixel-29-29." (Table 1)
- "conditional statements, and looping and branching structures allow the compact representation of solutions." (Generalized Planning)
- Inference: In/Out Dynamics = Capped (inferred) because pixel objects are enumerated from pixel-0-0 to pixel-29-29; Attention Dynamic = Dynamic (inferred) because planning programs use conditional, looping, and branching control.

---

## CSV Output (required)
Write a CSV file to "/home/jake/Developer/timeline/BIBLIOTHEQUE/03_COMP-REAS/Generalized Planning for the Abstraction and Reasoning Corpus (GPAR)/.TASK-DOMAINS.csv.tmp.4861b3076a2d4b8f9a38a13353523ea5" with the same rows and columns as the Task Table. Use the exact header:

task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic

Do not add extra columns, commentary, or blank lines.
