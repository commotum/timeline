# Transformer Binary Sort Prompt

You are classifying one paper folder into a strict binary label:
- `TRANSFORMER-YES`
- `TRANSFORMER-NO`

Input files:
- Primary source OCR markdown (expensive fallback): [SOURCE_MD_ABS_PATH]
- Hint file (cheap first pass): [HINT_TASK_DOMAINS_MD_ABS_PATH]
- Hint file (cheap first pass): [HINT_TASK_DOMAINS_CSV_ABS_PATH]
- Hint file (cheap first pass): [HINT_TASK_MODEL_RATIO_MD_ABS_PATH]
- Hint file (cheap first pass): [HINT_EXTENDING_DIMENSIONS_MD_ABS_PATH]

Output:
- Write markdown to: [MD_ABS_PATH]
- Paper folder: [OUTPUT_FOLDER]
- Paper stem: [FILE_STEM]

## Decision standard

Classify as `TRANSFORMER-YES` if the paper's central model (or a central hybrid model) materially uses Transformer-style self-attention (or clear variants such as ViT, Swin, Performer, RoFormer, GPT/BERT/LLaMA-style blocks, window/local/sparse/axial attention).

Classify as `TRANSFORMER-NO` if the central model does not materially use self-attention and Transformer mentions are only in related work, baselines, citations, or peripheral comparisons.

Important:
- Hybrid models count as `TRANSFORMER-YES` if self-attention is a core part of the model used for the main results.
- "Uncertain" is not allowed. You must pick YES or NO.
- You are the final decision maker for this binary sort.
- Token-frugality is required: do not open [SOURCE_MD_ABS_PATH] unless hint evidence is insufficient.

## Process (required)

### Pass 0: hint-first triage (required)
1. Examine hint files first.
2. If hints are sufficient for a high-confidence decision, finalize without opening [SOURCE_MD_ABS_PATH].

### Pass 1: source triage (only if needed)
1. If Pass 0 is not sufficient, scan [SOURCE_MD_ABS_PATH] for architecture cues (Transformer/attention/ViT/etc.) and anti-cues (RNN/CNN/Q-learning only, etc.).
2. If confidence becomes high, finalize.

### Pass 2: source deep dive (only if still needed)
1. If still unresolved after Pass 1, read method/model/architecture/training/appendix sections of [SOURCE_MD_ABS_PATH].
2. Resolve whether self-attention is in the actual model used for primary results.

## Output format (required)

# <Paper Title> (<Year or "Year not specified">)
Source: <filename>

## Binary decision
Decision: TRANSFORMER-YES
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

If the decision is NO, set:
Decision: TRANSFORMER-NO

Write the markdown to [MD_ABS_PATH]. Do not include extra sections.
