#!/usr/bin/env python3
"""
Iterative refinement loop for keyword-sort pipeline.

For each stage:
1) Run the stage.
2) Audit obvious title-level mistakes.
3) Investigate cause via existing evidence fields and apply stage-specific improvements.
4) Repeat until no new obvious issues or max iterations reached.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = ROOT / "run_keyword_pipeline.py"

STEP1_CSV = ROOT / "transformer_screen_results.csv"
STEP2_CSV = ROOT / "transformer_task_dimensions_results.csv"
STEP3_CSV = ROOT / "positional_encoding_results.csv"

RULES_JSON = ROOT / "iterative_refinement_rules.json"
REPORT_MD = ROOT / "iterative_refinement_report.md"


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("keyword_pipeline", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading pipeline module from {PIPELINE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def title_from_row(row: Dict[str, str]) -> str:
    return Path(row.get("paper_dir", "")).name


def append_note(row: Dict[str, str], note: str) -> None:
    prev = (row.get("notes") or "").strip()
    if not prev:
        row["notes"] = note
        return
    if note in prev:
        return
    row["notes"] = prev + ";" + note


@dataclass
class RefinementRules:
    stage1_force_yes_titles: List[str] = field(default_factory=list)
    stage1_force_no_titles: List[str] = field(default_factory=list)
    stage2_force_1d_titles: List[str] = field(default_factory=list)
    stage3_pe_overrides: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "stage1_force_yes_titles": sorted(set(self.stage1_force_yes_titles)),
            "stage1_force_no_titles": sorted(set(self.stage1_force_no_titles)),
            "stage2_force_1d_titles": sorted(set(self.stage2_force_1d_titles)),
            "stage3_pe_overrides": self.stage3_pe_overrides,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "RefinementRules":
        return cls(
            stage1_force_yes_titles=list(d.get("stage1_force_yes_titles", []) or []),
            stage1_force_no_titles=list(d.get("stage1_force_no_titles", []) or []),
            stage2_force_1d_titles=list(d.get("stage2_force_1d_titles", []) or []),
            stage3_pe_overrides=dict(d.get("stage3_pe_overrides", {}) or {}),
        )

    @classmethod
    def load(cls, path: Path) -> "RefinementRules":
        if not path.is_file():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)


# -----------------------------
# Stage 1
# -----------------------------


RL_TITLE_RE = re.compile(
    r"\b(reinforcement learning|double q-learning|q-learning|dqn|a3c|trpo|trust region policy optimization|policy optimization|actor-critic)\b",
    re.IGNORECASE,
)
TRANSFORMER_TITLE_RE = re.compile(
    r"\b(transformer|vision transformer|vit|vits|swin|roformer|bert|gpt|llama|performer|flashattention|llm|llms|large language model|large language models|deepseek|language agents?|long-context)\b",
    re.IGNORECASE,
)
NON_TRANSFORMER_LM_TITLE_RE = re.compile(
    r"\b(neural probabilistic language model|recurrent neural network based language model)\b",
    re.IGNORECASE,
)


def apply_stage1_overrides(rules: RefinementRules) -> int:
    fieldnames, rows = load_csv(STEP1_CSV)
    changed = 0
    if "notes" not in fieldnames:
        fieldnames.append("notes")

    force_yes = set(rules.stage1_force_yes_titles)
    force = set(rules.stage1_force_no_titles)

    for row in rows:
        t = title_from_row(row)
        if t not in force_yes:
            continue
        if row.get("label") != "transformer_yes":
            row["label"] = "transformer_yes"
            row["confidence"] = "high"
            ev = (row.get("evidence_lines") or "").strip()
            marker = "title_override_force_transformer_yes"
            row["evidence_lines"] = (ev + " | " + marker).strip(" | ")
            changed += 1
        append_note(row, "title_override_force_transformer_yes")

    for row in rows:
        t = title_from_row(row)
        if t not in force:
            continue
        if t in force_yes:
            continue
        if row.get("label") != "transformer_no":
            row["label"] = "transformer_no"
            row["confidence"] = "high"
            ev = (row.get("evidence_lines") or "").strip()
            marker = "title_override_force_transformer_no"
            row["evidence_lines"] = (ev + " | " + marker).strip(" | ")
            changed += 1
        append_note(row, "title_override_force_transformer_no")

    write_csv(STEP1_CSV, fieldnames, rows)
    return changed


def audit_stage1_titles(rules: RefinementRules) -> List[Dict[str, str]]:
    _, rows = load_csv(STEP1_CSV)
    issues: List[Dict[str, str]] = []

    forced_yes = set(rules.stage1_force_yes_titles)
    forced = set(rules.stage1_force_no_titles)
    for row in rows:
        title = title_from_row(row)
        low = title.lower()
        label = (row.get("label") or "").strip()
        try:
            a_hits = int((row.get("A_hits") or "0").strip() or 0)
        except Exception:
            a_hits = 0
        try:
            b_hits = int((row.get("B_hits") or "0").strip() or 0)
        except Exception:
            b_hits = 0

        if (
            label in {"transformer_yes", "hybrid_transformer_yes"}
            and title not in forced
            and RL_TITLE_RE.search(low)
            and not TRANSFORMER_TITLE_RE.search(low)
            and a_hits == 0
            and b_hits == 0
        ):
            issues.append(
                {
                    "kind": "force_no",
                    "title": title,
                    "paper_dir": row.get("paper_dir", ""),
                    "cause": "title looks RL-only but classified as transformer",
                    "evidence": row.get("evidence_lines", ""),
                }
            )
            continue

        if label in {"transformer_no", "uncertain"} and title not in forced_yes:
            if TRANSFORMER_TITLE_RE.search(low) and not NON_TRANSFORMER_LM_TITLE_RE.search(
                low
            ):
                issues.append(
                    {
                        "kind": "force_yes",
                        "title": title,
                        "paper_dir": row.get("paper_dir", ""),
                        "cause": "title has explicit transformer/LLM cues but is not in transformer set",
                        "evidence": row.get("evidence_lines", ""),
                    }
                )
    return issues


# -----------------------------
# Stage 2
# -----------------------------


LLM_TITLE_RE = re.compile(
    r"\b(language model|llm|gpt|bert|chinchilla|instructgpt|lora|qlora|megatron|toolformer|react|tree of thoughts|self-refine|realm|retrieval-augmented)\b",
    re.IGNORECASE,
)
SPATIAL_TITLE_RE = re.compile(
    r"\b(vision|image|video|point cloud|3d|4d|segment|segmentation|detection|arc[- ]?agi|maze|sudoku|multimodal|vlm|clip|autonomous driving)\b|^arc",
    re.IGNORECASE,
)


def parse_dims_cell(value: str) -> set[str]:
    out: set[str] = set()
    for tok in (value or "").split(";"):
        t = tok.strip()
        if t:
            out.add(t)
    return out


def apply_stage2_overrides(rules: RefinementRules) -> int:
    fieldnames, rows = load_csv(STEP2_CSV)
    changed = 0
    if "notes" not in fieldnames:
        fieldnames.append("notes")

    force = set(rules.stage2_force_1d_titles)
    for row in rows:
        title = title_from_row(row)
        if title not in force:
            continue
        task_csv_dims = parse_dims_cell(row.get("task_csv_dims", ""))
        # Respect TASK-DOMAINS.csv as primary evidence: do not downcast 2D+/multi-D to 1D by title.
        if any(d in {"2D", "3D", "4D"} for d in task_csv_dims):
            append_note(row, "title_override_force_1d_skipped_due_task_csv")
            continue
        if row.get("final_label") != "1D_only":
            row["final_label"] = "1D_only"
            row["final_dims"] = "1D"
            row["dimension_source"] = "title_override_force_1d"
            row["confidence"] = "high"
            changed += 1
        append_note(row, "title_override_force_1d")
    write_csv(STEP2_CSV, fieldnames, rows)
    return changed


def audit_stage2_titles(rules: RefinementRules) -> List[Dict[str, str]]:
    _, rows = load_csv(STEP2_CSV)
    issues: List[Dict[str, str]] = []
    forced = set(rules.stage2_force_1d_titles)

    for row in rows:
        final_label = row.get("final_label", "")
        if final_label not in {"2D_only", "3D_only", "4D_only", "multi-D"}:
            continue
        title = title_from_row(row)
        if title in forced:
            continue
        task_csv_dims = parse_dims_cell(row.get("task_csv_dims", ""))
        if any(d in {"2D", "3D", "4D"} for d in task_csv_dims):
            # Primary task-domain CSV already indicates 2D+ evidence.
            continue
        low = title.lower()
        if LLM_TITLE_RE.search(low) and not SPATIAL_TITLE_RE.search(low):
            issues.append(
                {
                    "title": title,
                    "paper_dir": row.get("paper_dir", ""),
                    "cause": "title looks 1D LLM-focused but classified 2D+/multi-D",
                    "evidence": row.get("task_evidence", ""),
                }
            )
    return issues


# -----------------------------
# Stage 3
# -----------------------------


KNOWN_PE_TITLE_PRIORS: Dict[str, Dict[str, str]] = {
    "ALBEF- Align Before Fuse": {
        "pe_label": "learned_absolute",
        "pe_components": "learned_absolute",
        "confidence": "medium",
        "notes": "title_prior_clip_vit_bert_family",
    },
    "BEiT- BERT Pre-Training of Image Transformers": {
        "pe_label": "learned_absolute",
        "pe_components": "learned_absolute",
        "confidence": "medium",
        "notes": "title_prior_backbone_beit",
    },
    "BLIP- Bootstrapping Language-Image Pre-training": {
        "pe_label": "learned_absolute",
        "pe_components": "learned_absolute",
        "confidence": "medium",
        "notes": "title_prior_clip_vit_bert_family",
    },
    "BLIP-2- Bootstrapping Language-Image Pre-training with Frozen Models": {
        "pe_label": "learned_absolute",
        "pe_components": "learned_absolute",
        "confidence": "medium",
        "notes": "title_prior_clip_vit_bert_family",
    },
    "Image as a Foreign Language- BEiT Pretraining for All Vision and Vision-Language Tasks": {
        "pe_label": "learned_absolute",
        "pe_components": "learned_absolute",
        "confidence": "medium",
        "notes": "title_prior_backbone_beit",
    },
    "Learning Transferable Visual Models From Natural Language Supervision": {
        "pe_label": "learned_absolute",
        "pe_components": "learned_absolute",
        "confidence": "medium",
        "notes": "title_prior_clip_vit_family",
    },
    "Masked Autoencoders Are Scalable Vision Learners (MAE)": {
        "pe_label": "sinusoidal_absolute",
        "pe_components": "sinusoidal_absolute",
        "confidence": "high",
        "notes": "title_prior_backbone_mae",
    },
    "Scaling Up Vision-Language Learning With Noisy Text Supervision (ALIGN)": {
        "pe_label": "learned_absolute",
        "pe_components": "learned_absolute",
        "confidence": "low",
        "notes": "title_prior_align_text_encoder",
    },
    "Sigmoid Loss for Language Image Pre-Training": {
        "pe_label": "learned_absolute",
        "pe_components": "learned_absolute",
        "confidence": "medium",
        "notes": "title_prior_siglip_vit_family",
    },
    "Training data-efficient image transformers & distillation through attention": {
        "pe_label": "learned_absolute",
        "pe_components": "learned_absolute",
        "confidence": "medium",
        "notes": "title_prior_deit_vit_family",
    },
}


def apply_stage3_overrides(rules: RefinementRules) -> int:
    fieldnames, rows = load_csv(STEP3_CSV)
    changed = 0
    if "notes" not in fieldnames:
        fieldnames.append("notes")

    for row in rows:
        title = title_from_row(row)
        ov = rules.stage3_pe_overrides.get(title)
        if not ov:
            continue
        if row.get("pe_label") != ov.get("pe_label", ""):
            row["pe_label"] = ov.get("pe_label", row.get("pe_label", ""))
            row["pe_components"] = ov.get("pe_components", row.get("pe_components", ""))
            row["confidence"] = ov.get("confidence", row.get("confidence", "medium"))
            changed += 1
        append_note(row, ov.get("notes", "title_prior_pe_override"))
    write_csv(STEP3_CSV, fieldnames, rows)
    return changed


def audit_stage3_titles(rules: RefinementRules) -> List[Dict[str, str]]:
    _, rows = load_csv(STEP3_CSV)
    issues: List[Dict[str, str]] = []
    existing = set(rules.stage3_pe_overrides.keys())
    for row in rows:
        if row.get("pe_label") != "unclear":
            continue
        title = title_from_row(row)
        if title in existing:
            continue
        if title in KNOWN_PE_TITLE_PRIORS:
            issues.append(
                {
                    "title": title,
                    "paper_dir": row.get("paper_dir", ""),
                    "cause": "known backbone/title prior for PE exists",
                    "evidence": row.get("ocr_evidence_lines", ""),
                }
            )
    return issues


# -----------------------------
# Orchestration
# -----------------------------


def run_stage1_loop(mod, rules: RefinementRules, max_iters: int, report: List[str]) -> None:
    report.append("## Stage 1 Loop")
    for i in range(1, max_iters + 1):
        mod.run_step1_transformer_screen(mod.list_paper_dirs())
        _ = apply_stage1_overrides(rules)
        issues = audit_stage1_titles(rules)
        report.append(f"- Iteration {i}: title-level issues found = {len(issues)}")
        if not issues:
            report.append("  - Stable.")
            break
        new_titles = []
        for it in issues:
            t = it["title"]
            if it.get("kind") == "force_yes":
                if t in rules.stage1_force_no_titles:
                    rules.stage1_force_no_titles.remove(t)
                if t not in rules.stage1_force_yes_titles:
                    rules.stage1_force_yes_titles.append(t)
                    new_titles.append(it)
            else:
                if t in rules.stage1_force_yes_titles:
                    rules.stage1_force_yes_titles.remove(t)
                if t not in rules.stage1_force_no_titles:
                    rules.stage1_force_no_titles.append(t)
                    new_titles.append(it)
        for it in new_titles:
            if it.get("kind") == "force_yes":
                report.append(
                    f"  - Added override: `{it['title']}` -> transformer_yes. Cause: {it['cause']}. Evidence: {it['evidence'][:180]}"
                )
            else:
                report.append(
                    f"  - Added override: `{it['title']}` -> transformer_no. Cause: {it['cause']}. Evidence: {it['evidence'][:180]}"
                )
        if not new_titles:
            report.append("  - No new overrides to add; stopping.")
            break


def run_stage2_loop(mod, rules: RefinementRules, max_iters: int, report: List[str]) -> None:
    report.append("## Stage 2 Loop")
    for i in range(1, max_iters + 1):
        mod.run_step2_task_dimensions()
        _ = apply_stage2_overrides(rules)
        issues = audit_stage2_titles(rules)
        report.append(f"- Iteration {i}: title-level issues found = {len(issues)}")
        if not issues:
            report.append("  - Stable.")
            break
        new_titles = []
        for it in issues:
            t = it["title"]
            if t not in rules.stage2_force_1d_titles:
                rules.stage2_force_1d_titles.append(t)
                new_titles.append(it)
        for it in new_titles:
            report.append(
                f"  - Added override: `{it['title']}` -> 1D_only. Cause: {it['cause']}. Evidence: {it['evidence'][:180]}"
            )
        if not new_titles:
            report.append("  - No new overrides to add; stopping.")
            break


def run_stage3_loop(mod, rules: RefinementRules, max_iters: int, report: List[str]) -> None:
    report.append("## Stage 3 Loop")
    for i in range(1, max_iters + 1):
        mod.run_step3_positional_encoding()
        _ = apply_stage3_overrides(rules)
        issues = audit_stage3_titles(rules)
        report.append(f"- Iteration {i}: title-level issues found = {len(issues)}")
        if not issues:
            report.append("  - Stable.")
            break
        new_titles = []
        for it in issues:
            t = it["title"]
            if t not in rules.stage3_pe_overrides:
                rules.stage3_pe_overrides[t] = KNOWN_PE_TITLE_PRIORS[t]
                new_titles.append(it)
        for it in new_titles:
            ov = rules.stage3_pe_overrides[it["title"]]
            report.append(
                f"  - Added PE prior override: `{it['title']}` -> {ov['pe_label']} ({ov['confidence']}). Cause: {it['cause']}"
            )
        if not new_titles:
            report.append("  - No new overrides to add; stopping.")
            break


def final_counts(report: List[str]) -> None:
    report.append("## Final Counts")
    for name, path, key in [
        ("Step1", STEP1_CSV, "label"),
        ("Step2", STEP2_CSV, "final_label"),
        ("Step3", STEP3_CSV, "pe_label"),
    ]:
        _, rows = load_csv(path)
        cnt: Dict[str, int] = {}
        for r in rows:
            k = r.get(key, "")
            cnt[k] = cnt.get(k, 0) + 1
        report.append(f"- {name}: total={len(rows)}")
        for k in sorted(cnt):
            report.append(f"  - {k}: {cnt[k]}")


def main(max_iters_per_stage: int = 4) -> None:
    mod = load_pipeline_module()
    rules = RefinementRules.load(RULES_JSON)
    # Keep overrides consistent: force-yes wins over force-no for the same title.
    rules.stage1_force_no_titles = sorted(
        set(rules.stage1_force_no_titles) - set(rules.stage1_force_yes_titles)
    )
    report: List[str] = [
        "# Iterative Refinement Report",
        "",
        "This run performs stage-by-stage loops:",
        "1. run stage",
        "2. audit obvious title-level mistakes",
        "3. investigate using current evidence fields and apply improvement overrides",
        "4. repeat until stable or max iterations",
        "",
    ]

    run_stage1_loop(mod, rules, max_iters_per_stage, report)
    run_stage2_loop(mod, rules, max_iters_per_stage, report)
    run_stage3_loop(mod, rules, max_iters_per_stage, report)

    # Re-run downstream stages once with finalized overrides to ensure consistency.
    mod.run_step1_transformer_screen(mod.list_paper_dirs())
    apply_stage1_overrides(rules)
    mod.run_step2_task_dimensions()
    apply_stage2_overrides(rules)
    mod.run_step3_positional_encoding()
    apply_stage3_overrides(rules)

    final_counts(report)
    rules.save(RULES_JSON)
    REPORT_MD.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Iterative refinement completed.")
    print(f"Rules: {RULES_JSON}")
    print(f"Report: {REPORT_MD}")


if __name__ == "__main__":
    main()
