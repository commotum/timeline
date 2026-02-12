# Positional Encoding Mechanism Sort Prompt

You are classifying one paper folder into exactly one positional-encoding (PE) label.

Allowed labels (exactly one):
- `PE-ROPE`
- `PE-LEARNED-ABSOLUTE`
- `PE-SINUSOIDAL-ABSOLUTE`
- `PE-RELATIVE-POSITION`
- `PE-ALIBI`
- `PE-MIXED`
- `PE-NONE-OR-IMPLICIT`
- `PE-INHERITED-DEFAULT`
- `PE-UNCLEAR`

Input files:
- Primary source OCR markdown (expensive fallback): [SOURCE_MD_ABS_PATH]
- Hint file (cheap first pass): [HINT_TRANSFORMER_YES_MD_ABS_PATH]
- Hint file (cheap first pass): [HINT_TRANSFORMER_NO_MD_ABS_PATH]
- Hint file (cheap first pass): [HINT_DIMENSION_1D_MD_ABS_PATH]
- Hint file (cheap first pass): [HINT_DIMENSION_2DPLUS_MD_ABS_PATH]
- Hint file (cheap first pass): [HINT_TASK_DOMAINS_MD_ABS_PATH]
- Hint file (cheap first pass): [HINT_TASK_DOMAINS_CSV_ABS_PATH]

Output:
- Write markdown to: [MD_ABS_PATH]
- Paper folder: [OUTPUT_FOLDER]
- Paper stem: [FILE_STEM]

## Decision standard

Primary rule:
- Prefer explicit evidence from the source paper text.

If explicit PE is missing, you MUST use informed judgment via model-family defaults (de-facto conventions) when reasonable, based on the paper's own architecture/backbone description. In this case choose `PE-INHERITED-DEFAULT` unless the default maps cleanly to one label with clear support and high confidence.

Examples of common defaults (use with caution, not blindly):
- Modern LLaMA-family: often RoPE.
- GPT-2 era decoder-only: often learned absolute embeddings.
- T5-family: often relative position bias.
- ViT/DeiT/BERT-style encoders: often learned absolute positional embeddings.
- Some vision pretraining variants (e.g., MAE encoder setups) may use fixed sin-cos.

Use these only when supported by architecture clues in the paper text and when no contradictory evidence is present.

Important:
- No "uncertain" outside label set; if still unresolved, choose `PE-UNCLEAR`.
- You are the final decision maker for this PE sort.
- Token-frugality is required: do not open [SOURCE_MD_ABS_PATH] unless hint evidence is insufficient.

## Process (required)

### Pass 0: hint-first triage (required)
1. Examine hint files first.
2. If hints plus inherited-default reasoning are sufficient, finalize without opening [SOURCE_MD_ABS_PATH].

### Pass 1: source triage (only if needed)
1. If Pass 0 is insufficient, scan [SOURCE_MD_ABS_PATH] for explicit PE terms: rotary/RoPE, relative, ALiBi, sinusoidal, learned positional embeddings, etc.
2. If confidence becomes high, finalize.

### Pass 2: source deep dive (only if still needed)
1. If still unresolved after Pass 1, read model/method/architecture/implementation/appendix sections in [SOURCE_MD_ABS_PATH].
2. Determine explicit PE if possible.

### Pass 3: inherited-default reasoning (required when explicit PE remains missing)
1. Infer likely PE from explicitly mentioned backbone/model family and era.
2. If this is a prior/default assumption rather than directly stated PE, choose:
   - `PE-INHERITED-DEFAULT` (preferred), or
   - a concrete label only if strongly justified.
3. Mark confidence accordingly (usually low/medium) and document the assumption.

Hints policy:
- Hint files are valid evidence for `hint-only` decisions.
- If source file was read, include at least one source quote.

## Output format (required)

# <Paper Title> (<Year or "Year not specified">)
Source: <filename>

## PE decision
Decision: PE-ROPE
Confidence: <high|medium|low>
Basis: <hint-only|source-triage|source-deep-dive|inherited-default|mixed>

## Why
- <reason 1>
- <reason 2>

## Evidence
- "<verbatim quote>" (Section/Page/Line context)
- "<verbatim quote>" (Section/Page/Line context)
If Basis is `hint-only`, evidence may come from hint files and must name the file.

## Inherited-default reasoning
- Used: <yes|no>
- If yes: <what default/prior was used and why>

## Pass accounting
Pass 0 (hint-first): <performed> - <result>
Pass 1 (source triage): <performed|skipped> - <reason>
Pass 2 (source deep dive): <performed|skipped> - <reason>
Pass 3 (inherited-default): <performed|skipped> - <reason>

Write the markdown to [MD_ABS_PATH]. Do not include extra sections.
