# Plan 3: Positional Encoding Extraction for 2D+/Multi-D Transformer Papers

## Pipeline Position
This is step 3 of the pipeline:
1. Is it a Transformer?
2. Is it 2D/3D/4D or multi-D?
3. What positional encoding method is used?

## Goal
From papers already labeled as Transformer-family and 2D+/multi-D, determine the positional encoding (PE) approach used by the model.

Target PE labels include:
- `learned_absolute`
- `sinusoidal_absolute`
- `rope`
- `axial_rope`
- `learned_rope`
- `relative_position` (including bias-style)
- `alibi`
- `mixed` (multiple PE schemes in the main model)
- `none_or_implicit`
- `unclear`

## Inputs
- `RUNNERS/keyword-sort/transformer_task_dimensions_results.csv`
  - use rows where `final_label` is one of:
    - `2D_only`
    - `3D_only`
    - `4D_only`
    - `multi-D`
- OCR paper markdown as primary source:
  - `<paper_dir>/<basename(paper_dir)>.md`
- Confirmation files:
  - `<paper_dir>/TASK-DOMAINS.md`
  - `<paper_dir>/TASK-DOMAINS.csv`

## Core Constraints
- Use `.md` and `.csv` only.
- Do not open PDFs or images.
- Keep token usage low: prefer keyword extraction + short context windows over full-document reads.

## Why This Step Is Hard
PE is often described:
- in appendix, implementation, or ablation sections (not abstract),
- with overloaded terms (`position`, `embedding`, `bias`) that cause false positives,
- with hybrid setups (for example space PE != time PE),
- as optional toggles at finetuning time.

So this step uses section-aware, evidence-scored keyword extraction rather than simple raw hit counting.

## PE Keyword Bank (Robust)

### A) Direct PE Terms (highest priority)
- `positional encoding`
- `position encoding`
- `positional embedding`
- `position embedding`
- `absolute positional encoding`
- `absolute position embedding`
- `APE`
- `relative positional encoding`
- `relative position encoding`
- `relative position bias`
- `positional bias`

### B) RoPE Family Terms
- `RoPE`
- `rotary`
- `rotary position embedding`
- `rotary positional embedding`
- `axial RoPE`
- `2D RoPE`
- `3D RoPE`
- `learnable RoPE`
- `learned RoPE`
- `interpolated RoPE`
- `rope scaling`
- `NTK-aware RoPE`
- `xPos`

### C) Other Named Variants
- `sinusoidal`
- `Fourier features`
- `Fourier positional encoding`
- `Gaussian Fourier`
- `ALiBi`
- `T5-style relative`
- `disentangled position`
- `convolutional positional encoding`
- `conditional positional encoding`

### D) Context Cues That Increase Confidence
- `we use`
- `we employ`
- `our model uses`
- `is performed using`
- `in our architecture`
- `processor block`
- `encoder`
- `decoder`
- `finetuning`
- `ablation`

### E) False-Positive Zones (down-weight)
- references/bibliography
- related work only
- baseline descriptions not used in final model

## Section-Aware Strategy (Low Token)

### Pass 1: Build 2D+/Multi-D Transformer Paper List
Read `transformer_task_dimensions_results.csv` and keep only:
- `2D_only`, `3D_only`, `4D_only`, `multi-D`

Output:
- `RUNNERS/keyword-sort/pe_candidate_paper_dirs.txt`
- `RUNNERS/keyword-sort/pe_candidate_ocr_files.txt`

### Pass 2: Wide PE Keyword Sweep
Run `rg` once per candidate OCR file with broad PE regex.
Save all hits with line numbers:
- `RUNNERS/keyword-sort/pe_hits_raw.tsv`

Regex should include all groups above and case variants.

### Pass 3: Context Window Extraction Around Hits
For each hit line, extract small windows (`line-4` to `line+4`) only.
Save:
- `RUNNERS/keyword-sort/pe_hit_windows.tsv`

This is the main anti-token-burn tactic: only inspect neighborhoods around candidate PE lines.

### Pass 4: Evidence Scoring Per Paper
For each paper, compute scores:
- `direct_pe_score` from A hits in method/architecture/training sections
- `rope_score` from B hits
- `other_variant_score` from C hits
- `context_bonus` from D cues near hits
- `false_positive_penalty` if hits are reference-only or related-work-only

Then apply label rules (below).

### Stage 3.5: Inheritance Recovery for Remaining `unclear`
After Pass 4, run a second-stage recovery pass only on papers still labeled `unclear`.

Purpose:
- Recover PE labels for papers that do not restate PE explicitly but clearly inherit from known backbones or are wrappers around base models.

#### 3.5A) Paper-type split
Assign each unclear paper one type:
- `backbone_derived` (explicitly built on a named base model/backbone)
- `method_wrapper` (reasoning/agent/training/runtime method applied to an existing model)
- `infra_system` (optimization/systems paper; usually PE-agnostic)
- `unknown`

#### 3.5B) Inheritance cue patterns
Search for phrases like:
- `initialized from`
- `pretrained ... backbone`
- `frozen backbone`
- `we use <MODEL>`
- `based on <MODEL>`
- `adopt ... architecture`
- `using CLIP/ViT/BERT/GPT/T5/LLaMA/...`

Require at least one model/backbone identity plus one inheritance cue.

#### 3.5C) Backbone-to-PE prior map (use only when explicit PE is absent)
- `ViT`, `DeiT`, `CLIP ViT` -> usually `learned_absolute` (vision side)
- `Swin` -> usually `relative_position`
- `BERT`, `GPT-2` -> usually `learned_absolute`
- `T5` -> `relative_position` (bias style)
- `LLaMA`, `RoFormer` -> `rope`
- `MAE` -> `sinusoidal_absolute` (2D fixed)

If a paper clearly combines text + vision backbones with different defaults:
- label `mixed` and list components.

#### 3.5D) Wrapper/system handling rules
- If paper is a wrapper over an explicit base LM/VLM and does not modify PE:
  - set PE by inheritance
  - set `pe_source = inherited_backbone`
- If paper is infra-only and has no model-level PE claims:
  - keep `unclear` or set `not_applicable` (optional policy)
  - set `pe_source = unknown_or_na`

#### 3.5E) Confidence policy for Stage 3.5
- `high`: explicit PE in paper text
- `medium`: inherited from explicitly named backbone
- `low`: weak inferred inheritance

Stage 3.5 should never overwrite a `high`-confidence explicit label.

## Label Decision Rules

### Rule 1: `axial_rope`
If explicit `axial RoPE` or equivalent appears in model/architecture context.

### Rule 2: `learned_rope`
If RoPE appears with `learnable`, `trainable`, `learned`, `finetuning adaptation` context.

### Rule 3: `rope`
If RoPE/rotary appears clearly in core model and Rules 1-2 do not apply.

### Rule 4: `learned_absolute`
If explicit learned position embeddings / APE are used in core model.

### Rule 5: `sinusoidal_absolute`
If sinusoidal PE is explicit in core model.

### Rule 6: `relative_position`
If relative position encoding/bias is explicit and no stronger RoPE/APE label dominates.

### Rule 7: `alibi`
If ALiBi explicitly used in core model.

### Rule 8: `mixed`
If two or more PE families are explicitly used in the model
for different axes/stages/components (for example space uses RoPE, time uses relative; or RoPE + APE).

### Rule 9: `none_or_implicit`
If text explicitly states no positional encoding / implicit positional handling.

### Rule 10: `unclear`
If only weak or ambiguous evidence remains.

## Conflict Resolution
- If OCR says one PE and TASK files imply another:
  - prefer OCR method/appendix evidence.
  - keep task-file hints in notes.
- If PE appears only in baseline sections:
  - do not assign that PE to the paper's core model.
- If PE changes between pretraining and finetuning:
  - set `mixed` and record stage-specific details.

## Output Files
Create:
- `RUNNERS/keyword-sort/positional_encoding_results.csv`
- `RUNNERS/keyword-sort/positional_encoding_summary.md`
- `RUNNERS/keyword-sort/positional_encoding_uncertain.md`

Suggested CSV schema:

```csv
paper_dir,dimension_label,pe_label,pe_components,pe_source,stage_scope,confidence,ocr_evidence_lines,taskfile_evidence_lines,notes
```

Where:
- `dimension_label` is from step-2 output (`2D_only`/`3D_only`/`4D_only`/`multi-D`)
- `pe_label` is one of the labels above
- `pe_components` can store multiple methods like `axial_rope;relative_position;learned_absolute`
- `pe_source` in `{explicit,inherited_backbone,method_wrapper,unknown_or_na}`
- `stage_scope` in `{pretrain,finetune,both,unspecified}`

## Confidence Definition
- `high`: explicit architecture statement with direct PE method naming.
- `medium`: strong keyword + context signal but slight ambiguity.
- `low`: weak or indirect evidence; likely for manual review.

## Minimal-Token Command Skeleton

Build candidate list:

```bash
python - <<'PY'
import csv
src = "RUNNERS/keyword-sort/transformer_task_dimensions_results.csv"
out = "RUNNERS/keyword-sort/pe_candidate_paper_dirs.txt"
keep = {"2d_only","3d_only","4d_only","multi-d"}
with open(src, newline="", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    r = csv.DictReader(f)
    for row in r:
        lbl = (row.get("final_label") or "").strip().lower()
        p = (row.get("paper_dir") or "").strip()
        if lbl in keep and p:
            g.write(p + "\n")
PY
```

PE sweep:

```bash
while IFS= read -r d; do
  b="$(basename "$d")"
  f="$d/$b.md"
  [[ -f "$f" ]] || continue
  rg -n -i -H "positional encoding|position encoding|positional embedding|position embedding|absolute position|\\bAPE\\b|relative position|position bias|\\bRoPE\\b|rotary|axial rope|learnable rope|learned rope|sinusoidal|fourier positional|\\bALiBi\\b|t5-style relative|convolutional positional|conditional positional" "$f"
done < RUNNERS/keyword-sort/pe_candidate_paper_dirs.txt > RUNNERS/keyword-sort/pe_hits_raw.tsv
```

## Manual Review Slice (Only for Uncertain)
For rows labeled `unclear` or `low` confidence:
- inspect only top 3 strongest hit windows per paper.
- do not read entire files unless still unresolved.
- In Stage 3.5, inspect only top backbone-inheritance lines (not full papers) before assigning inherited labels.

## Quality Checks
- No `pe_label` assigned without at least one OCR evidence line.
- `mixed` requires explicit evidence for at least two PE families.
- Papers with only baseline PE mentions are not mislabeled.
- At least 10% random spot-check across non-unclear labels.
- Stage 3.5 inferred labels must include explicit inheritance evidence and set `pe_source` accordingly.

## Expected Outcome
A high-precision map of PE methods across Transformer papers focused on 2D+ tasks, enabling direct analysis of PE heterogeneity and lack of consensus.
