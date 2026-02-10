# Synthesis From Examples: Interaction Models and Algorithms (Not specified in the paper.)
Source: Synthesis From Examples- Interaction Models and Algorithms.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Program synthesis (bitvector algorithm manipulation) | input-output examples of bitvectors | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | bitvector program (straight-line bitwise instructions) | 1D (t) (inferred) | Open (inferred) |
| Program synthesis (syntactic string transformation in spreadsheets) | input-output examples of spreadsheet strings | 1D (t) (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | syntactic string transformation script | 1D (t) (inferred) | Open (inferred) |
| Program synthesis (number transformation in spreadsheets) | input-output examples of spreadsheet number strings | 1D (t) (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | number transformation script | 1D (t) (inferred) | Open (inferred) |
| Program synthesis (semantic string transformation with table lookup) | input-output examples of strings plus relational background tables | 1D (t); 2D (x, y) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | semantic string transformation script | 1D (t) (inferred) | Open (inferred) |
| Program synthesis (table layout transformation in spreadsheets) | input-output examples of tables | 2D (x, y) (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | table transformation script | 1D (t) (inferred) | Open (inferred) |
| Program synthesis (ruler/compass geometry constructions) | random geometry models/examples over points, lines, and circles | 2D (x, y) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | geometry construction program | 1D (t) (inferred) | Open (inferred) |
| Problem generation (algebraic proof problems) | example algebraic proof problem(s) | 1D (t); 2D (x, y) (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | new algebraic proof problems | 1D (t); 2D (x, y) (inferred) | Open (inferred) |
| Prediction (mathematical term completion) | mathematical text prefixes/sessions | 1D (t) (inferred) | Open (inferred) | Not specified in the paper. | Not specified in the paper. | predicted mathematical sub-terms | 1D (t) (inferred) | Open (inferred) |
| Prediction (repetitive drawing completion) | partial sketches or initial drawing objects | 2D (x, y) (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | predicted repetitive drawing objects | 2D (x, y) (inferred) | Open (inferred) |

## Summary
The paper covers a broad synthesis-from-examples portfolio: bitvector program synthesis, spreadsheet macro synthesis (strings, numbers, semantic lookups, tables), geometry construction synthesis, algebraic problem generation, and predictive interfaces for math terms and drawings. Inputs span mainly 1D symbolic sequences and 2D tabular/geometric structures, with outputs including synthesized programs and predicted/generated artifacts. The text supports mostly Open dynamics (example sets and generated artifacts can grow), while several synthesis workflows justify Constructed state through explicit search/template data structures. Attention behavior is explicitly dynamic only where the paper describes distinguishing-input generation, lookup-based selection, or goal-directed search; otherwise it is not specified.

## Evidence
### Task: Program synthesis (bitvector algorithm manipulation)
- "Bitvector algorithms are typically straight-line sequence of instructions" (Section IV-A)
- "for synthesis of a bitvector algorithm (§IV-A) from input-output examples" (Figure 1 caption, Section II-B)
- "[14] describes a constraint solving based (§III-C) inductive synthesizer" (Section IV-A)
- Inference: `In Dimension = 1D (t)` from "input bitvector" and bit-string examples (Section IV-A, Figure 1); `In Dynamics = Open` and `Attention Dynamic = Dynamic` from "generates a new input in each round" and repeated example addition (Section II-B); `State Dynamic = Constructed` from explicit candidate-program search over sets of consistent programs (Section II-B).

### Task: Program synthesis (syntactic string transformation in spreadsheets)
- "(a) Syntactic String Transformation" (Figure 2, Section IV-B)
- "The DSL for *Syntactic string transformations* [26] includes substring and concatenate operators" (Section IV-B)
- "generate scripts for automating repetitive tasks from input-output examples" (Section IV-B)
- Inference: `In Dimension = 1D (t)` from string inputs/outputs in Figure 2(a); `In Dynamics = Open` and `Out Dynamics = Open` from example-driven synthesis with no explicit bound; `State Dynamic = Constructed` from version-space representation/manipulation of consistent artifacts (Sections III-A and IV-B).

### Task: Program synthesis (number transformation in spreadsheets)
- "(b) Number Transformation" (Figure 2, Section IV-B)
- "*Number transformations* [28] allow for formatting and rounding transformations on numbers" (Section IV-B)
- "generate scripts for automating repetitive tasks from input-output examples" (Section IV-B)
- Inference: `In Dimension = 1D (t)` from number strings like "0d 5h 26m" in Figure 2(b); `In/Out Dynamics = Open` because bounds are not specified; `State Dynamic = Constructed` from version-space based inductive synthesis (Sections III-A and IV-B).

### Task: Program synthesis (semantic string transformation with table lookup)
- "(c) Semantic String Transformation" and "(d) Semantic String Transformation" (Figure 2, Section IV-B)
- "combine syntactic transformations with lookup operations from other relational tables" (Section IV-B)
- "Background Knowledge | (user-defined) Tables" (Figure 2(d), Section IV-B)
- Inference: `In Dimension = 1D (t); 2D (x, y)` from string examples plus relational tables; `Attention Dynamic = Dynamic` from runtime lookup operations over tables; `State Dynamic = Constructed` from version-space/data-structure based synthesis (Sections III-A and IV-B).

### Task: Program synthesis (table layout transformation in spreadsheets)
- "(e) Table Transformation" (Figure 2, Section IV-B)
- "*Table transformations* [29] allow for layout transformations on tables" (Section IV-B)
- "can automate the tasks in (a), (b), (c)/(d), and (e)" (Figure 2 caption, Section IV-B)
- Inference: `In Dimension = 2D (x, y)` from input/output tables in Figure 2(e); `In/Out Dynamics = Open` because table-size bounds are not specified; `State Dynamic = Constructed` from version-space based script synthesis (Sections III-A and IV-B).

### Task: Program synthesis (ruler/compass geometry constructions)
- "Geometry constructions are essentially straight-line programs" (Section IV-C)
- "manipulate geometry objects (points, lines, and circles)" (Section IV-C)
- "geometric constructions can be synthesized from random examples or models" (Section IV-C)
- Inference: `In Dimension = 2D (x, y)` from coordinate examples like "p1 = <81.62, 99.62>" (Figure 3, Section IV-C); `Attention Dynamic = Dynamic` from "goal-directed heuristics"; `State Dynamic = Constructed` from brute-force search over construction candidates (Sections III-B and IV-C).

### Task: Problem generation (algebraic proof problems)
- "Generating fresh problems that involve using a given set of concepts" (Section IV-D)
- "Synthesis of algebraic proof problems" (Figure 4 caption, Section IV-D)
- "automatically synthesized starting from a given example problem" (Section IV-D)
- Inference: `In/Out Dimension = 1D (t); 2D (x, y)` from equation-style terms and matrix/determinant forms shown in Figure 4; `State Dynamic = Constructed` because "a generalized problem template is synthesized from the example problem(s)" and validated by "testing on random inputs (§II-C)" (Section IV-D).

### Task: Prediction (mathematical term completion)
- "Synthesis of low-entropy mathematical terms from their prefixes" (Figure 5 caption, Section IV-E)
- "Predicting sub-terms that the user is likely to input next" (Section IV-E)
- "can be phrased as a synthesis-from-example problem [34]" (Section IV-E)
- Inference: `In/Out Dimension = 1D (t)` from textual prefixes/sub-terms; `In/Out Dynamics = Open` from multi-expression "sessions" without an explicit bound (Section IV-E).

### Task: Prediction (repetitive drawing completion)
- "Synthesis of repetitive geometric drawings from partial sketches" (Figure 6 caption, Section IV-F)
- "prediction of other objects from the initial beautified objects" (Figure 6 caption, Section IV-F)
- "Predicting the repetitive objects in a drawing from few examples" (Section IV-F)
- Inference: `In/Out Dimension = 2D (x, y)` from geometric drawing objects; `State Dynamic = Constructed` from "synthesizing object transformation logics"; `In/Out Dynamics = Open` because object repetition count is not bounded in the text (Figure 6 caption, Section IV-F).
