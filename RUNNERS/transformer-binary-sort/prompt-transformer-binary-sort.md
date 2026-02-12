# Transformer Binary Sort Prompt

You are classifying one paper folder into a strict binary label:
- `TRANSFORMER-YES`
- `TRANSFORMER-NO`

Input files:
- Paper OCR markdown: [SOURCE_MD_ABS_PATH]
  - Use this file to read the paper abstract first.
- Task/domain analysis markdown: [TASK_DOMAINS_MD_ABS_PATH]
  - Expected contents: task framing, domain tags, and model-family cues.
  - Read this file in full.
- Task/domain analysis CSV: [TASK_DOMAINS_CSV_ABS_PATH]
  - Expected contents: structured rows/fields for task and model indicators.
  - Read this file in full.
- Model-ratio analysis markdown: [TASK_MODEL_RATIO_MD_ABS_PATH]
  - Expected contents: model-family breakdown and attention/Transformer signals.
  - Read this file in full.
- Extending-dimensions analysis markdown: [EXTENDING_DIMENSIONS_MD_ABS_PATH]
  - Expected contents: architecture-focused notes, including whether self-attention is central.
  - Read this file in full.

Output:
- Write markdown to: [MD_ABS_PATH]
- Paper folder: [OUTPUT_FOLDER]
- Paper stem: [FILE_STEM]

Path handling:
- If any bracket variable resolves to `MISSING`, treat that file as unavailable, skip it, and state that it was unavailable in your reasoning/evidence.

## Decision standard

Classify as `TRANSFORMER-YES` if the paper's central model (or a central hybrid model) materially uses Transformer-style self-attention (or clear variants such as ViT, Swin, Performer, RoFormer, GPT/BERT/LLaMA-style blocks, window/local/sparse/axial attention).

Classify as `TRANSFORMER-NO` if the central model does not materially use self-attention and Transformer mentions are only in related work, baselines, citations, or peripheral comparisons.

Important:
- Hybrid models count as `TRANSFORMER-YES` if self-attention is a core part of the model used for the main results.
- "Uncertain" is not allowed. You must pick YES or NO.
- You are the final decision maker for this binary sort.
- Default policy: make the decision from the abstract plus the four auxiliary files.
- Do not read the full paper body unless the abstract + auxiliary files are genuinely insufficient.

## Process (required)

### Pass 1: abstract + auxiliary files (required)
1. Read the abstract in [SOURCE_MD_ABS_PATH].
2. Read [TASK_DOMAINS_MD_ABS_PATH], [TASK_DOMAINS_CSV_ABS_PATH], [TASK_MODEL_RATIO_MD_ABS_PATH], and [EXTENDING_DIMENSIONS_MD_ABS_PATH] in full when available.
3. If this evidence is sufficient for a high-confidence decision, finalize without reading the rest of the paper.

### Pass 2: targeted source scan (only if needed)
1. If Pass 1 is not sufficient, scan [SOURCE_MD_ABS_PATH] for architecture cues (Transformer/attention/ViT/etc.) and anti-cues (RNN/CNN/Q-learning only, etc.), focusing on model/method sections.
2. Only read beyond the abstract when Pass 1 remains unclear, and read only the additional sections needed to finalize.
3. If confidence becomes high, finalize.

## Output format (required)

# <Paper Title> (<Year or "Year not specified">)
Source: <filename>

## Binary decision
Decision: TRANSFORMER-YES
Confidence: <high|medium|low>
Basis: <abstract-aux-only|source-targeted-scan>

## Why
- <reason 1>
- <reason 2>

## Evidence
- "<verbatim quote>" (Section/Page/Line context)
- "<verbatim quote>" (Section/Page/Line context)
If Basis is `abstract-aux-only`, evidence may come from the abstract and auxiliary files and must name the file.

## Pass accounting
Pass 1 (abstract + auxiliary files): <performed> - <result>
Pass 2 (targeted source scan): <performed|skipped> - <reason>

If the decision is NO, set:
Decision: TRANSFORMER-NO

Write the markdown to [MD_ABS_PATH]. Do not include extra sections.
