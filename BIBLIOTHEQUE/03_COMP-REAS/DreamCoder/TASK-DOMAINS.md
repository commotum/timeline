# DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning (Not specified in the paper.)
Source: DreamCoder.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Program synthesis (list processing) | input/output examples of list transformations (lists of numbers) | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | list-processing program | Not specified in the paper. | Not specified in the paper. |
| Program synthesis (text editing) | input/output examples of text strings | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text-editing program | Not specified in the paper. | Not specified in the paper. |
| Program synthesis (LOGO graphics drawing) | target images to be drawn | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | graphics program controlling a pen | Not specified in the paper. | Not specified in the paper. |
| Program synthesis (tower building plan) | image of a tower and block locations | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | program/plan for a simulated hand to build the tower | Not specified in the paper. | Not specified in the paper. |
| Program induction (probabilistic regex) | example strings (small number; 5 per concept) | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | probabilistic regular expression | Not specified in the paper. | Not specified in the paper. |
| Symbolic regression (parametric equations) | real-valued data from curves/trajectories | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | parametric equation/program | Not specified in the paper. | Not specified in the paper. |
| Equation discovery (physical laws) | numerical examples of physical-law data (vectors as lists) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | equations describing physical laws/identities | Not specified in the paper. | Not specified in the paper. |
| Program synthesis (recursive list routines) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | programs for recursive list routines | Not specified in the paper. | Not specified in the paper. |

## Summary
DreamCoder is evaluated on program induction tasks spanning list processing and text editing from input/output examples, visual drawing and tower-building from images, probabilistic regex induction from strings, symbolic regression, physics-law equation discovery, and recursive list routine synthesis. The inputs cover 1D sequences (lists, strings, trajectories) and 2D images/locations, while outputs are programs or equations. The paper explicitly fixes the number of examples in some domains (list processing and regex), but leaves other dynamics and attention/state characteristics unspecified.

## Evidence
### Task: Program synthesis (list processing)
- "list processing and text editing." (Section: Results)
- "tasks specified by a conditional mapping (i.e., input/output examples)." (Section: Results)
- "each with 15 input/output examples." (Section: Results)
- "sort lists of numbers" (Section: Results)
- Inference: In Dimension=1D (t) because the task uses "lists of numbers"; In Dynamics=Fixed because tasks are "each with 15 input/output examples." (Section: Results)

### Task: Program synthesis (text editing)
- "Synthesizing programs that edit text" (Section: Results)
- "see the mapping \"Alan Turing\" → \"A.T.\"" (Section: Results)
- "infer a program that transforms \"Grace Hopper\" to \"G.H.\"" (Section: Results)
- Inference: In Dimension=1D (t) because inputs are text strings (e.g., "Alan Turing" → "A.T."). (Section: Results)

### Task: Program synthesis (LOGO graphics drawing)
- "drawing a corpus of 160 images" (Section: Results)
- "control over a 'pen'" (Section: Results)
- "writes programs controlling a 'pen' that draws the target picture." (Figure 4 caption)
- Inference: In Dimension=2D (x, y) because the task input is an "image"/"target picture." (Section: Results; Figure 4 caption)

### Task: Program synthesis (tower building plan)
- "observes both an image of a tower" (Section: Results)
- "the locations of each of its blocks" (Section: Results)
- "must write a program that plans" (Section: Results)
- Inference: In Dimension=2D (x, y) because inputs include an "image" and spatial "locations" of blocks. (Section: Results)

### Task: Program induction (probabilistic regex)
- "inferring a probabilistic regular expression" (Section: Results)
- "from a small number of strings" (Section: Results)
- "5 example strings per concept." (Section: Results)
- Inference: In Dimension=1D (t) because inputs are "strings"; In Dynamics=Fixed because there are "5 example strings per concept." (Section: Results)

### Task: Symbolic regression (parametric equations)
- "inferring real-valued parametric equations" (Section: Results)
- "generating smooth trajectories" (Section: Results)
- "Each task is to fit data generated by a specific curve" (Section: Results)
- Inference: In Dimension=1D (t) because the input is a "trajectory"/"curve" (1D index). (Section: Results)

### Task: Equation discovery (physical laws)
- "learning equations describing 60 different physical laws" (Section: From learning libraries to learning languages)
- "based on numerical examples of data obeying each equation." (Section: From learning libraries to learning languages)
- "Vectors are represented as lists of numbers." (Figure 7 caption)
- Inference: In Dimension=1D (t) because inputs are "lists of numbers" (vectors). (Figure 7 caption)

### Task: Program synthesis (recursive list routines)
- "asked it to solve 20 basic programming tasks" (Section: From learning libraries to learning languages)
- "Learning a language for recursive list routines" (Figure 7 caption)
