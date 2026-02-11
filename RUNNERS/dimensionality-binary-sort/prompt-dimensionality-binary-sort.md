# Dimensionality Binary Sort Prompt

You are classifying one paper folder into a strict binary label:
- `DIMENSION-1D`
- `DIMENSION-2DPLUS`

Input files:
- Primary source OCR markdown (expensive fallback): [SOURCE_MD_ABS_PATH]
- Hint file (cheap first pass): [HINT_TASK_DOMAINS_MD_ABS_PATH]
- Hint file (cheap first pass): [HINT_TASK_DOMAINS_CSV_ABS_PATH]
- Hint file (cheap first pass): [HINT_TASK_MODEL_RATIO_MD_ABS_PATH]
- Hint file (cheap first pass): [HINT_TRANSFORMER_YES_MD_ABS_PATH]
- Hint file (cheap first pass): [HINT_TRANSFORMER_NO_MD_ABS_PATH]

Output:
- Write markdown to: [MD_ABS_PATH]
- Paper folder: [OUTPUT_FOLDER]
- Paper stem: [FILE_STEM]

## Decision standard

Classify as `DIMENSION-1D` if the core evaluated tasks are exclusively 1D sequence/time/token style (e.g., text tokens, time series, autoregressive sequence modeling) with no central 2D/3D/4D task requirement.

Classify as `DIMENSION-2DPLUS` if any core task in the paper materially involves 2D/3D/4D or mixed-dimensional tasks (including image/video/point-cloud/grid/spatiotemporal tasks), even when the paper also includes 1D tasks.

Important:
- If a paper contains both 1D and 2D+ core tasks, choose `DIMENSION-2DPLUS`.
- "Uncertain" is not allowed. You must pick one of the two labels.
- You are the final decision maker for this binary sort.
- Token-frugality is required: do not open [SOURCE_MD_ABS_PATH] unless hint evidence is insufficient.

## Process (required)

### Pass 0: hint-first triage (required)
1. Examine hint files first, especially `TASK-DOMAINS.csv` and `TASK-DOMAINS.md`.
2. If hints are sufficient for high-confidence decision, finalize without opening [SOURCE_MD_ABS_PATH].

### Pass 1: source triage (only if needed)
1. If Pass 0 is insufficient, scan [SOURCE_MD_ABS_PATH] for dimensional cues (1D sequence/time vs 2D/3D/4D/spatial-temporal cues).
2. If confidence becomes high, finalize.

### Pass 2: source deep dive (only if still needed)
1. If still unresolved after Pass 1, read task/problem setup, datasets, method/application sections, and appendices in [SOURCE_MD_ABS_PATH].
2. Resolve whether the core tasks require only 1D or include any 2D+ workload.

## Output format (required)

# <Paper Title> (<Year or "Year not specified">)
Source: <filename>

## Binary decision
Decision: DIMENSION-1D
Confidence: <high|medium|low>
Basis: <hint-only|source-triage|source-deep-dive>

## Why
- <reason 1>
- <reason 2>

## Evidence
- "<verbatim quote>" (Section/Page/Line context)
- "<verbatim quote>" (Section/Page/Line context)
If Basis is `hint-only`, evidence may come from hint files and must name the file.

## Pass accounting
Pass 0 (hint-first): <performed> - <result>
Pass 1 (source triage): <performed|skipped> - <reason>
Pass 2 (source deep dive): <performed|skipped> - <reason>

If the decision is 2D+, set:
Decision: DIMENSION-2DPLUS

Write the markdown to [MD_ABS_PATH]. Do not include extra sections.
