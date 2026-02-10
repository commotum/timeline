#!/usr/bin/env python3
"""
Execute the keyword-sort pipeline end-to-end:
1) Transformer-family screening
2) Task dimension classification for transformer papers
3) Positional encoding classification for 2D+/multi-D transformer papers
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUT_DIR = SCRIPT_DIR

TARGET_CLASS_DIRS = [
    REPO_ROOT / "BIBLIOTHEQUE" / "03_COMP-REAS",
    REPO_ROOT / "BIBLIOTHEQUE" / "05_ML-FNDTNS",
]


# -----------------------------
# Shared helpers
# -----------------------------


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_lines(path: Path) -> List[str]:
    return read_text(path).splitlines()


def clip(text: str, limit: int = 220) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def find_ocr_path(paper_dir: Path) -> Path:
    return paper_dir / f"{paper_dir.name}.md"


def list_paper_dirs() -> List[Path]:
    out: List[Path] = []
    for class_dir in TARGET_CLASS_DIRS:
        if not class_dir.is_dir():
            continue
        for p in sorted(class_dir.iterdir(), key=lambda x: x.name.lower()):
            if p.is_dir():
                out.append(p)
    return out


def class_code_from_dir(paper_dir: Path) -> str:
    parent = paper_dir.parent.name
    if "_" in parent:
        return parent.split("_", 1)[1]
    return parent


def write_tsv(path: Path, rows: Iterable[Tuple[str, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write("\t".join(row))
            f.write("\n")


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


@dataclass
class LineContext:
    line_no: int
    text: str
    heading: str
    in_references: bool
    in_related_work: bool


def build_line_contexts(lines: List[str]) -> List[LineContext]:
    contexts: List[LineContext] = []
    heading = ""
    in_refs = False
    in_related = False

    heading_re = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
    all_caps_heading_re = re.compile(r"^\s{0,4}[0-9A-Z][0-9A-Z .:_/-]{6,}\s*$")

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        m = heading_re.match(line)

        if m:
            heading = m.group(1).strip()
            h = heading.lower()
            if "reference" in h or h == "bibliography":
                in_refs = True
            elif "appendix" in h:
                # Keep refs true if references already started; appendices can be before refs.
                in_refs = in_refs and ("reference" in h)
            in_related = "related work" in h
        elif all_caps_heading_re.match(line):
            heading = stripped
            h = heading.lower()
            if "reference" in h or h == "bibliography":
                in_refs = True
            in_related = "related work" in h

        # OCR files sometimes use plain "References" with no markdown heading.
        if not in_refs and re.match(r"^\s*references?\s*$", stripped, flags=re.IGNORECASE):
            in_refs = True

        contexts.append(
            LineContext(
                line_no=i,
                text=line,
                heading=heading,
                in_references=in_refs,
                in_related_work=in_related,
            )
        )

    return contexts


def window_text(lines: List[str], line_no: int, radius: int = 2) -> str:
    start = max(1, line_no - radius)
    end = min(len(lines), line_no + radius)
    return "\n".join(lines[start - 1 : end])


# -----------------------------
# Step 1: Transformer screening
# -----------------------------


A_RE = re.compile(
    r"\b(transformers?|vision transformer|encoder[- ]only transformer|decoder[- ]only transformer|encoder[- ]decoder transformer|vit\b|gpt\b|bert\b|roformer|swin transformer)\b",
    flags=re.IGNORECASE,
)
B_RE = re.compile(
    r"\b(self[- ]attention|multi[- ]head attention|scaled dot[- ]product attention|cross[- ]attention|windowed attention|local attention|sparse attention|axial attention|hierarchical attention|causal attention|flash ?attention)\b",
    flags=re.IGNORECASE,
)
C_RE = re.compile(
    r"\b(tokenization|tokenized|token\b|context length|positional encoding|positional embedding|rotary|rope|relative position|absolute position embedding|\bAPE\b|\bQKV\b)\b",
    flags=re.IGNORECASE,
)
D_RE = re.compile(
    r"\b(recurrent neural network|\bRNN\b|\bLSTM\b|\bGRU\b|convolutional neural network|\bCNN\b|policy gradient|q[- ]?learning|actor[- ]critic|\bPPO\b|\bDQN\b|\bSARSA\b)\b",
    flags=re.IGNORECASE,
)

CONTEXT_RE = re.compile(
    r"\b(we use|we employ|our model|our architecture|architecture|method|model uses|implemented|is performed using|processor block|encoder|decoder|training|finetuning)\b",
    flags=re.IGNORECASE,
)

BASELINE_RE = re.compile(
    r"\b(baseline|related work|prior work|compared with|comparison)\b",
    flags=re.IGNORECASE,
)


@dataclass
class Hit:
    line_no: int
    text: str
    trusted: bool
    contextful: bool
    baseline_like: bool


def scan_transformer_hits(ocr_path: Path) -> Dict[str, List[Hit]]:
    lines = read_lines(ocr_path)
    contexts = build_line_contexts(lines)

    out: Dict[str, List[Hit]] = {"A": [], "B": [], "C": [], "D": []}

    for ctx in contexts:
        line = ctx.text
        win = window_text(lines, ctx.line_no, radius=2)
        contextful = bool(CONTEXT_RE.search(win))
        baseline_like = bool(BASELINE_RE.search(win)) or ctx.in_related_work
        trusted = not ctx.in_references

        for key, rex in (("A", A_RE), ("B", B_RE), ("C", C_RE), ("D", D_RE)):
            if rex.search(line):
                out[key].append(
                    Hit(
                        line_no=ctx.line_no,
                        text=line,
                        trusted=trusted,
                        contextful=contextful,
                        baseline_like=baseline_like,
                    )
                )
    return out


def run_step1_transformer_screen(paper_dirs: List[Path]) -> None:
    paper_ocr_list = OUT_DIR / "paper_ocr_list.txt"
    hits_a_file = OUT_DIR / "hits_A_strong.tsv"
    hits_b_file = OUT_DIR / "hits_B_attention.tsv"
    hits_c_file = OUT_DIR / "hits_C_support.tsv"
    hits_d_file = OUT_DIR / "hits_D_classic.tsv"
    results_csv = OUT_DIR / "transformer_screen_results.csv"
    summary_md = OUT_DIR / "transformer_screen_summary.md"

    ocr_paths: List[Path] = []
    rows: List[Dict[str, str]] = []
    hit_rows = {"A": [], "B": [], "C": [], "D": []}

    for paper_dir in paper_dirs:
        ocr = find_ocr_path(paper_dir)
        if ocr.is_file():
            ocr_paths.append(ocr)
        else:
            rows.append(
                {
                    "paper_dir": str(paper_dir.relative_to(REPO_ROOT)),
                    "class_code": class_code_from_dir(paper_dir),
                    "A_hits": "0",
                    "B_hits": "0",
                    "C_hits": "0",
                    "D_hits": "0",
                    "label": "uncertain",
                    "confidence": "low",
                    "evidence_lines": "missing OCR markdown",
                }
            )
            continue

        hits = scan_transformer_hits(ocr)

        # Trusted hit counts
        a_hits = [h for h in hits["A"] if h.trusted]
        b_hits = [h for h in hits["B"] if h.trusted]
        c_hits = [h for h in hits["C"] if h.trusted]
        d_hits = [h for h in hits["D"] if h.trusted]

        # Context-weighted hits
        b_context_hits = [
            h for h in b_hits if h.contextful and not h.baseline_like
        ]
        d_context_hits = [
            h for h in d_hits if h.contextful and not h.baseline_like
        ]

        transformer_signal = False
        if len(a_hits) >= 1:
            transformer_signal = True
        elif len(b_hits) >= 2 and len(c_hits) >= 2 and len(b_context_hits) >= 1:
            transformer_signal = True

        if transformer_signal:
            if len(d_context_hits) >= 1:
                label = "hybrid_transformer_yes"
            else:
                label = "transformer_yes"
        else:
            if len(a_hits) == 0 and len(b_hits) == 0 and len(d_hits) >= 2:
                label = "transformer_no"
            else:
                label = "uncertain"

        if label in {"transformer_yes", "hybrid_transformer_yes"}:
            confidence = "high" if len(a_hits) >= 1 else "medium"
        elif label == "transformer_no":
            confidence = "medium"
        else:
            confidence = "low"

        evidence_parts: List[str] = []
        for group_name, group_hits in (("A", a_hits), ("B", b_hits), ("D", d_hits)):
            if group_hits:
                h = group_hits[0]
                evidence_parts.append(
                    f"{group_name}@{h.line_no}: {clip(h.text, 140)}"
                )
        evidence = " | ".join(evidence_parts) if evidence_parts else ""

        rows.append(
            {
                "paper_dir": str(paper_dir.relative_to(REPO_ROOT)),
                "class_code": class_code_from_dir(paper_dir),
                "A_hits": str(len(a_hits)),
                "B_hits": str(len(b_hits)),
                "C_hits": str(len(c_hits)),
                "D_hits": str(len(d_hits)),
                "label": label,
                "confidence": confidence,
                "evidence_lines": evidence,
            }
        )

        rel_ocr = str(ocr.relative_to(REPO_ROOT))
        for key in ("A", "B", "C", "D"):
            for h in hits[key]:
                hit_rows[key].append(
                    (rel_ocr, str(h.line_no), clip(h.text, 400))
                )

    rows.sort(key=lambda r: r["paper_dir"].lower())

    paper_ocr_list.write_text(
        "\n".join(str(p.relative_to(REPO_ROOT)) for p in sorted(ocr_paths)) + "\n",
        encoding="utf-8",
    )
    write_tsv(hits_a_file, hit_rows["A"])
    write_tsv(hits_b_file, hit_rows["B"])
    write_tsv(hits_c_file, hit_rows["C"])
    write_tsv(hits_d_file, hit_rows["D"])

    write_csv(
        results_csv,
        rows,
        [
            "paper_dir",
            "class_code",
            "A_hits",
            "B_hits",
            "C_hits",
            "D_hits",
            "label",
            "confidence",
            "evidence_lines",
        ],
    )

    cnt = Counter(r["label"] for r in rows)
    yes_cnt = cnt["transformer_yes"] + cnt["hybrid_transformer_yes"]
    summary = [
        "# Transformer Screen Summary",
        "",
        f"Total papers scanned: {len(rows)}",
        f"Transformer-family (yes+hybrid): {yes_cnt}",
        f"- transformer_yes: {cnt['transformer_yes']}",
        f"- hybrid_transformer_yes: {cnt['hybrid_transformer_yes']}",
        f"- transformer_no: {cnt['transformer_no']}",
        f"- uncertain: {cnt['uncertain']}",
        "",
        "Files generated:",
        "- transformer_screen_results.csv",
        "- hits_A_strong.tsv",
        "- hits_B_attention.tsv",
        "- hits_C_support.tsv",
        "- hits_D_classic.tsv",
        "- paper_ocr_list.txt",
    ]
    summary_md.write_text("\n".join(summary) + "\n", encoding="utf-8")


# -----------------------------
# Step 2: Task dimensions
# -----------------------------


DIM_1D_RE = re.compile(
    r"\b(1d|one[- ]dimensional|language model(ing)?|machine translation|text generation|autoregressive text|speech recognition|time series)\b",
    flags=re.IGNORECASE,
)
DIM_2D_RE = re.compile(
    r"\b(2d|two[- ]dimensional|image|pixel|patch|grid|table|board|maze|sudoku|\(x,\s*y\))\b",
    flags=re.IGNORECASE,
)
DIM_3D_RE = re.compile(
    r"\b(3d|three[- ]dimensional|point cloud|voxel|volume|mesh|video|\(x,\s*y,\s*z\)|\(x,\s*y,\s*t\))\b",
    flags=re.IGNORECASE,
)
DIM_4D_RE = re.compile(
    r"\b(4d|four[- ]dimensional|\(x,\s*y,\s*z,\s*t\)|spatiotemporal volume|space[- ]time volume)\b",
    flags=re.IGNORECASE,
)
TASK_DIM_RE = re.compile(r"\b([1-4]D)\b")


def scan_dims_in_ocr(ocr_path: Path) -> Tuple[Set[str], Dict[str, List[Tuple[int, str]]]]:
    lines = read_lines(ocr_path)
    contexts = build_line_contexts(lines)
    dims: Set[str] = set()
    evid: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for ctx in contexts:
        if ctx.in_references or ctx.in_related_work:
            continue
        text = ctx.text
        if DIM_4D_RE.search(text):
            dims.add("4D")
            evid["4D"].append((ctx.line_no, clip(text)))
        if DIM_3D_RE.search(text):
            dims.add("3D")
            evid["3D"].append((ctx.line_no, clip(text)))
        if DIM_2D_RE.search(text):
            dims.add("2D")
            evid["2D"].append((ctx.line_no, clip(text)))
        if DIM_1D_RE.search(text):
            dims.add("1D")
            evid["1D"].append((ctx.line_no, clip(text)))
    return dims, evid


def scan_dims_in_task_files(task_csv: Path, task_md: Path) -> Tuple[Set[str], List[str]]:
    dims: Set[str] = set()
    evidence: List[str] = []

    for p in (task_csv, task_md):
        if not p.is_file():
            continue
        txt = read_text(p)
        for m in TASK_DIM_RE.finditer(txt):
            dims.add(m.group(1).upper())
        # Keep up to a few lines that mention D labels
        for line_no, line in enumerate(txt.splitlines(), start=1):
            if TASK_DIM_RE.search(line):
                evidence.append(f"{p.name}:{line_no}:{clip(line, 160)}")
                if len(evidence) >= 5:
                    break

    return dims, evidence


def run_step2_task_dimensions() -> None:
    in_csv = OUT_DIR / "transformer_screen_results.csv"
    if not in_csv.is_file():
        raise FileNotFoundError(f"Missing required input: {in_csv}")

    out_csv = OUT_DIR / "transformer_task_dimensions_results.csv"
    summary_md = OUT_DIR / "transformer_task_dimensions_summary.md"
    paper_dirs_txt = OUT_DIR / "transformer_paper_dirs.txt"
    ocr_list_txt = OUT_DIR / "transformer_ocr_files.txt"
    task_csv_list_txt = OUT_DIR / "transformer_taskdomain_csv_files.txt"
    task_md_list_txt = OUT_DIR / "transformer_taskdomain_md_files.txt"
    missing_ocr_txt = OUT_DIR / "transformer_missing_ocr.txt"

    hits_dim_1d = OUT_DIR / "hits_dim_1d.tsv"
    hits_dim_2d = OUT_DIR / "hits_dim_2d.tsv"
    hits_dim_3d = OUT_DIR / "hits_dim_3d.tsv"
    hits_dim_4d = OUT_DIR / "hits_dim_4d.tsv"
    hits_taskdims = OUT_DIR / "hits_taskdomains_dim.tsv"

    step1_rows = list(csv.DictReader(in_csv.open("r", encoding="utf-8", newline="")))
    keep_labels = {"transformer_yes", "hybrid_transformer_yes"}

    selected = [r for r in step1_rows if (r.get("label") or "").strip() in keep_labels]

    rows: List[Dict[str, str]] = []
    paper_dir_lines: List[str] = []
    ocr_lines: List[str] = []
    task_csv_lines: List[str] = []
    task_md_lines: List[str] = []
    missing_ocr_lines: List[str] = []

    hits_rows = {"1D": [], "2D": [], "3D": [], "4D": [], "TASK": []}

    for row in selected:
        rel_dir = row["paper_dir"]
        paper_dir = REPO_ROOT / rel_dir
        paper_dir_lines.append(rel_dir)

        ocr = find_ocr_path(paper_dir)
        task_csv = paper_dir / "TASK-DOMAINS.csv"
        task_md = paper_dir / "TASK-DOMAINS.md"

        if task_csv.is_file():
            task_csv_lines.append(str(task_csv.relative_to(REPO_ROOT)))
        if task_md.is_file():
            task_md_lines.append(str(task_md.relative_to(REPO_ROOT)))

        if not ocr.is_file():
            missing_ocr_lines.append(rel_dir)
            rows.append(
                {
                    "paper_dir": rel_dir,
                    "step1_label": row["label"],
                    "final_label": "uncertain",
                    "final_dims": "",
                    "ocr_dims": "",
                    "task_dims": "",
                    "confidence": "low",
                    "ocr_evidence": "missing OCR markdown",
                    "task_evidence": "",
                }
            )
            continue

        ocr_lines.append(str(ocr.relative_to(REPO_ROOT)))
        ocr_dims, ocr_evid = scan_dims_in_ocr(ocr)
        task_dims, task_evid = scan_dims_in_task_files(task_csv, task_md)

        # Promote non-1D task dims from confirmation files when OCR already indicates 2D+
        final_dims: Set[str] = set()
        if ocr_dims:
            final_dims |= ocr_dims
            if final_dims.intersection({"2D", "3D", "4D"}):
                final_dims |= {d for d in task_dims if d in {"2D", "3D", "4D"}}
                if "1D" in task_dims and "1D" in ocr_dims:
                    final_dims.add("1D")
            else:
                final_dims |= task_dims
        else:
            final_dims |= task_dims

        # Guardrail: avoid OCR-only 1D contamination when the paper is clearly 2D+.
        if (
            "1D" in final_dims
            and final_dims.intersection({"2D", "3D", "4D"})
            and "1D" not in task_dims
        ):
            final_dims.discard("1D")

        if not final_dims:
            final_label = "uncertain"
        elif len(final_dims) == 1:
            only = next(iter(final_dims))
            final_label = f"{only}_only"
        else:
            final_label = "multi-D"

        if ocr_dims and task_dims:
            confidence = "high" if (ocr_dims & task_dims) else "medium"
        elif ocr_dims and not task_dims:
            confidence = "medium"
        elif task_dims and not ocr_dims:
            confidence = "low"
        else:
            confidence = "low"

        def top_evidence(evid_map: Dict[str, List[Tuple[int, str]]]) -> str:
            parts: List[str] = []
            for d in ("1D", "2D", "3D", "4D"):
                if d in evid_map and evid_map[d]:
                    ln, tx = evid_map[d][0]
                    parts.append(f"{d}@{ln}:{tx}")
            return " | ".join(parts)

        ocr_evidence = top_evidence(ocr_evid)
        task_evidence = " | ".join(task_evid[:3])

        rows.append(
            {
                "paper_dir": rel_dir,
                "step1_label": row["label"],
                "final_label": final_label,
                "final_dims": ";".join(sorted(final_dims)),
                "ocr_dims": ";".join(sorted(ocr_dims)),
                "task_dims": ";".join(sorted(task_dims)),
                "confidence": confidence,
                "ocr_evidence": ocr_evidence,
                "task_evidence": task_evidence,
            }
        )

        # Write hit summaries for traceability
        rel_ocr = str(ocr.relative_to(REPO_ROOT))
        for dim, evids in ocr_evid.items():
            for ln, tx in evids[:30]:
                hits_rows[dim].append((rel_ocr, str(ln), tx))
        for ev in task_evid[:30]:
            parts = ev.split(":", 2)
            if len(parts) == 3:
                hits_rows["TASK"].append((f"{rel_dir}/{parts[0]}", parts[1], parts[2]))

    rows.sort(key=lambda r: r["paper_dir"].lower())

    write_csv(
        out_csv,
        rows,
        [
            "paper_dir",
            "step1_label",
            "final_label",
            "final_dims",
            "ocr_dims",
            "task_dims",
            "confidence",
            "ocr_evidence",
            "task_evidence",
        ],
    )

    paper_dirs_txt.write_text("\n".join(sorted(paper_dir_lines)) + "\n", encoding="utf-8")
    ocr_list_txt.write_text("\n".join(sorted(ocr_lines)) + "\n", encoding="utf-8")
    task_csv_list_txt.write_text("\n".join(sorted(task_csv_lines)) + "\n", encoding="utf-8")
    task_md_list_txt.write_text("\n".join(sorted(task_md_lines)) + "\n", encoding="utf-8")
    missing_ocr_txt.write_text("\n".join(sorted(missing_ocr_lines)) + "\n", encoding="utf-8")

    write_tsv(hits_dim_1d, hits_rows["1D"])
    write_tsv(hits_dim_2d, hits_rows["2D"])
    write_tsv(hits_dim_3d, hits_rows["3D"])
    write_tsv(hits_dim_4d, hits_rows["4D"])
    write_tsv(hits_taskdims, hits_rows["TASK"])

    cnt = Counter(r["final_label"] for r in rows)
    summary = [
        "# Transformer Task Dimensions Summary",
        "",
        f"Transformer papers from step 1: {len(rows)}",
        "",
        "Final labels:",
    ]
    for k in sorted(cnt):
        summary.append(f"- {k}: {cnt[k]}")
    summary += [
        "",
        "Files generated:",
        "- transformer_task_dimensions_results.csv",
        "- transformer_paper_dirs.txt",
        "- transformer_ocr_files.txt",
        "- transformer_taskdomain_csv_files.txt",
        "- transformer_taskdomain_md_files.txt",
        "- transformer_missing_ocr.txt",
        "- hits_dim_1d.tsv / hits_dim_2d.tsv / hits_dim_3d.tsv / hits_dim_4d.tsv",
        "- hits_taskdomains_dim.tsv",
    ]
    summary_md.write_text("\n".join(summary) + "\n", encoding="utf-8")


# -----------------------------
# Step 3: Positional encoding
# -----------------------------


PE_PATTERNS = {
    "axial_rope": re.compile(r"\b(axial\s+rope|axial\s+rotary|rotary.*axial)\b", re.IGNORECASE),
    "learned_rope": re.compile(
        r"\b((learnable|learned|trainable)\s+(rope|rotary)|(rope|rotary).{0,25}(learnable|learned|trainable))\b",
        re.IGNORECASE,
    ),
    "rope": re.compile(
        r"\b(rope|rotary positional? embedding|rotary position embedding|rotary)\b",
        re.IGNORECASE,
    ),
    "learned_absolute": re.compile(
        r"\b(learned (absolute )?(position|positional) embedding|absolute position embedding|\bAPE\b)\b",
        re.IGNORECASE,
    ),
    "sinusoidal_absolute": re.compile(
        r"\b(sinusoidal (position|positional) (encoding|embedding)|sinusoidal embeddings?)\b",
        re.IGNORECASE,
    ),
    "relative_position": re.compile(
        r"\b(relative (position|positional) (encoding|embedding|bias)|position bias|t5-style relative)\b",
        re.IGNORECASE,
    ),
    "alibi": re.compile(r"\bALiBi\b", re.IGNORECASE),
    "other_variant": re.compile(
        r"\b(fourier positional|fourier features|gaussian fourier|xpos|disentangled position|convolutional positional|conditional positional)\b",
        re.IGNORECASE,
    ),
    "none_or_implicit": re.compile(
        r"\b(no positional encoding|without positional encoding|implicit positional|position[- ]free)\b",
        re.IGNORECASE,
    ),
}

PE_CONTEXT_RE = re.compile(
    r"\b(we use|we employ|our model|our architecture|architecture|model uses|is performed using|processor block|encoder|decoder|training|finetuning|appendix|implementation)\b",
    re.IGNORECASE,
)

PE_BASELINE_RE = re.compile(
    r"\b(baseline|related work|prior work|comparison|compared to)\b",
    re.IGNORECASE,
)

PRETRAIN_RE = re.compile(r"\b(pretrain|pre-training|pretrained)\b", re.IGNORECASE)
FINETUNE_RE = re.compile(r"\b(finetune|fine-tune|finetuning|fine-tuning)\b", re.IGNORECASE)

PE_METHOD_HEADING_RE = re.compile(
    r"\b(architecture|method|model|implementation|details|training|appendix|experiments?)\b",
    re.IGNORECASE,
)

# Fallback (broader) patterns for unclear cases only.
PE_FALLBACK_PATTERNS = {
    "axial_rope": re.compile(r"\baxial\b.{0,35}\b(rope|rotary)\b", re.IGNORECASE),
    "learned_rope": re.compile(
        r"\b(learnable|learned|trainable)\b.{0,35}\b(rope|rotary)\b|\b(rope|rotary)\b.{0,35}\b(learnable|learned|trainable)\b",
        re.IGNORECASE,
    ),
    "rope": re.compile(r"\b(rope|rotary)\b", re.IGNORECASE),
    "learned_absolute": re.compile(
        r"\b(learnable|learned|trainable)\b.{0,45}\b(position|positional)\b.{0,45}\b(embedding|encoding)\b|\b(position|positional)\b.{0,30}\b(embedding|encoding)\b.{0,30}\b(learnable|learned|trainable)\b|\bAPE\b",
        re.IGNORECASE,
    ),
    "sinusoidal_absolute": re.compile(
        r"\b(sinusoidal|sine|cosine)\b.{0,45}\b(position|positional|embedding|encoding)\b",
        re.IGNORECASE,
    ),
    "relative_position": re.compile(
        r"\b(relative|t5-style)\b.{0,30}\b(position|positional|bias|encoding|embedding)\b|\bRPE\b",
        re.IGNORECASE,
    ),
    "alibi": re.compile(r"\bALiBi\b", re.IGNORECASE),
    "other_variant": re.compile(
        r"\b(coordinate|coords?|spatial|temporal|fourier)\b.{0,45}\b(embedding|encoding|features?|bias)\b",
        re.IGNORECASE,
    ),
    "none_or_implicit": re.compile(
        r"\b(no positional encoding|without positional encoding|implicit positional|position[- ]free)\b",
        re.IGNORECASE,
    ),
}

PE_FALLBACK_SUPPORT_RE = re.compile(
    r"\b(position|positional|embedding|encoding|bias|coordinate|spatial|temporal|sinusoidal|fourier|rope|rotary|alibi|relative)\b",
    re.IGNORECASE,
)


def pe_group(family: str) -> str:
    if family in {"axial_rope", "learned_rope", "rope"}:
        return "rope_family"
    if family in {"learned_absolute", "sinusoidal_absolute"}:
        return "absolute_family"
    if family == "relative_position":
        return "relative_family"
    if family == "alibi":
        return "alibi_family"
    if family == "other_variant":
        return "other_family"
    if family == "none_or_implicit":
        return "none_family"
    return family


def choose_rope_subtype(scores: Dict[str, int]) -> str:
    if scores.get("axial_rope", 0) > 0:
        return "axial_rope"
    if scores.get("learned_rope", 0) > 0:
        return "learned_rope"
    return "rope"


def classify_pe(scores: Dict[str, int]) -> Tuple[str, List[str]]:
    group_scores: Dict[str, int] = defaultdict(int)
    for fam, sc in scores.items():
        group_scores[pe_group(fam)] += sc

    non_none_groups = [
        g for g, sc in group_scores.items() if g != "none_family" and sc >= 2
    ]

    components: List[str] = []
    if scores.get("axial_rope", 0) > 0:
        components.append("axial_rope")
    elif scores.get("learned_rope", 0) > 0:
        components.append("learned_rope")
    elif scores.get("rope", 0) > 0:
        components.append("rope")

    for fam in (
        "learned_absolute",
        "sinusoidal_absolute",
        "relative_position",
        "alibi",
        "other_variant",
        "none_or_implicit",
    ):
        if scores.get(fam, 0) > 0:
            components.append(fam)

    # Mixed if multiple non-none PE families are meaningfully present
    if len(non_none_groups) >= 2:
        return "mixed", components

    # Single-family selection
    if scores.get("axial_rope", 0) > 0:
        return "axial_rope", components
    if scores.get("learned_rope", 0) > 0:
        return "learned_rope", components
    if scores.get("rope", 0) > 0:
        return "rope", components
    if scores.get("learned_absolute", 0) > 0:
        return "learned_absolute", components
    if scores.get("sinusoidal_absolute", 0) > 0:
        return "sinusoidal_absolute", components
    if scores.get("relative_position", 0) > 0:
        return "relative_position", components
    if scores.get("alibi", 0) > 0:
        return "alibi", components
    if scores.get("other_variant", 0) > 0:
        return "other_variant", components
    if scores.get("none_or_implicit", 0) > 0:
        return "none_or_implicit", components
    return "unclear", components


def fallback_pe_scores(
    lines: List[str], contexts: List[LineContext]
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    scores: Dict[str, int] = defaultdict(int)
    evidence: Dict[str, List[str]] = defaultdict(list)

    for ctx in contexts:
        if ctx.in_references:
            continue

        line = ctx.text
        if not PE_FALLBACK_SUPPORT_RE.search(line):
            continue

        win = window_text(lines, ctx.line_no, radius=4)
        contextful = bool(PE_CONTEXT_RE.search(win))
        method_heading = bool(PE_METHOD_HEADING_RE.search(ctx.heading or ""))
        baseline_like = bool(PE_BASELINE_RE.search(win)) or ctx.in_related_work

        for fam, rex in PE_FALLBACK_PATTERNS.items():
            if not rex.search(line):
                continue

            # Require some method context for broad matches.
            if baseline_like and not (contextful or method_heading):
                continue

            if method_heading and contextful and not baseline_like:
                weight = 2
            elif contextful and not baseline_like:
                weight = 1
            elif method_heading and not baseline_like:
                weight = 1
            else:
                # Keep very weak traces out to avoid false positives.
                weight = 0

            if weight <= 0:
                continue

            scores[fam] += weight
            if len(evidence[fam]) < 3:
                evidence[fam].append(f"{ctx.line_no}:{clip(line, 160)}")

    return scores, evidence


def run_step3_positional_encoding() -> None:
    in_csv = OUT_DIR / "transformer_task_dimensions_results.csv"
    if not in_csv.is_file():
        raise FileNotFoundError(f"Missing required input: {in_csv}")

    out_csv = OUT_DIR / "positional_encoding_results.csv"
    summary_md = OUT_DIR / "positional_encoding_summary.md"
    uncertain_md = OUT_DIR / "positional_encoding_uncertain.md"
    candidate_dirs_txt = OUT_DIR / "pe_candidate_paper_dirs.txt"
    candidate_ocr_txt = OUT_DIR / "pe_candidate_ocr_files.txt"
    hits_raw_tsv = OUT_DIR / "pe_hits_raw.tsv"
    hit_windows_tsv = OUT_DIR / "pe_hit_windows.tsv"

    rows_in = list(csv.DictReader(in_csv.open("r", encoding="utf-8", newline="")))
    keep_dims = {"2D_only", "3D_only", "4D_only", "multi-D"}
    candidates = [r for r in rows_in if (r.get("final_label") or "").strip() in keep_dims]

    candidate_dirs_txt.write_text(
        "\n".join(sorted(r["paper_dir"] for r in candidates)) + "\n",
        encoding="utf-8",
    )

    out_rows: List[Dict[str, str]] = []
    raw_hits_rows: List[Tuple[str, str, str, str]] = []
    window_rows: List[Tuple[str, str, str, str]] = []
    candidate_ocr_lines: List[str] = []

    for row in candidates:
        rel_dir = row["paper_dir"]
        paper_dir = REPO_ROOT / rel_dir
        ocr = find_ocr_path(paper_dir)
        task_md = paper_dir / "TASK-DOMAINS.md"
        task_csv = paper_dir / "TASK-DOMAINS.csv"

        if not ocr.is_file():
            out_rows.append(
                {
                    "paper_dir": rel_dir,
                    "dimension_label": row["final_label"],
                    "pe_label": "unclear",
                    "pe_components": "",
                    "stage_scope": "unspecified",
                    "confidence": "low",
                    "ocr_evidence_lines": "missing OCR markdown",
                    "taskfile_evidence_lines": "",
                    "notes": "",
                }
            )
            continue

        candidate_ocr_lines.append(str(ocr.relative_to(REPO_ROOT)))
        lines = read_lines(ocr)
        contexts = build_line_contexts(lines)

        family_scores: Dict[str, int] = defaultdict(int)
        family_evidence: Dict[str, List[str]] = defaultdict(list)
        stage_mentions = {"pretrain": 0, "finetune": 0}

        for ctx in contexts:
            if ctx.in_references:
                continue
            line = ctx.text
            win = window_text(lines, ctx.line_no, radius=4)
            contextful = bool(PE_CONTEXT_RE.search(win))
            baseline_like = bool(PE_BASELINE_RE.search(win)) or ctx.in_related_work

            if PRETRAIN_RE.search(win):
                stage_mentions["pretrain"] += 1
            if FINETUNE_RE.search(win):
                stage_mentions["finetune"] += 1

            for fam, rex in PE_PATTERNS.items():
                if rex.search(line):
                    # Weighting: strong in context, weak in baseline context
                    if contextful and not baseline_like:
                        weight = 2
                    elif baseline_like and not contextful:
                        weight = 0
                    else:
                        weight = 1

                    family_scores[fam] += weight

                    rel_ocr = str(ocr.relative_to(REPO_ROOT))
                    raw_hits_rows.append(
                        (rel_ocr, str(ctx.line_no), fam, clip(line, 320))
                    )
                    window_rows.append(
                        (
                            rel_ocr,
                            str(ctx.line_no),
                            fam,
                            clip(win.replace("\n", " || "), 700),
                        )
                    )
                    if len(family_evidence[fam]) < 4:
                        family_evidence[fam].append(f"{ctx.line_no}:{clip(line, 160)}")

        pe_label, components = classify_pe(family_scores)
        fallback_used = False
        if pe_label == "unclear":
            fb_scores, fb_evidence = fallback_pe_scores(lines, contexts)
            if sum(fb_scores.values()) > 0:
                pe_label_fb, components_fb = classify_pe(fb_scores)
                if pe_label_fb != "unclear":
                    pe_label = pe_label_fb
                    components = components_fb
                    fallback_used = True
                    for fam, sc in fb_scores.items():
                        family_scores[fam] += sc
                    for fam, evs in fb_evidence.items():
                        for ev in evs:
                            if ev not in family_evidence[fam] and len(family_evidence[fam]) < 4:
                                family_evidence[fam].append(ev)

        total_score = sum(family_scores.values())
        if pe_label == "unclear":
            confidence = "low"
        elif fallback_used:
            confidence = "medium" if total_score >= 2 else "low"
        elif total_score >= 6:
            confidence = "high"
        elif total_score >= 2:
            confidence = "medium"
        else:
            confidence = "low"

        if stage_mentions["pretrain"] > 0 and stage_mentions["finetune"] > 0:
            stage_scope = "both"
        elif stage_mentions["finetune"] > 0:
            stage_scope = "finetune"
        elif stage_mentions["pretrain"] > 0:
            stage_scope = "pretrain"
        else:
            stage_scope = "unspecified"

        # Task-file confirmation hints (light-touch)
        task_hints: List[str] = []
        for p in (task_md, task_csv):
            if not p.is_file():
                continue
            for i, ln in enumerate(read_lines(p), start=1):
                if re.search(
                    r"positional|position|rope|rotary|relative|sinusoidal|alibi|ape",
                    ln,
                    flags=re.IGNORECASE,
                ):
                    task_hints.append(f"{p.name}:{i}:{clip(ln, 140)}")
                    if len(task_hints) >= 3:
                        break

        ocr_evidence_lines: List[str] = []
        for fam in (
            "axial_rope",
            "learned_rope",
            "rope",
            "learned_absolute",
            "sinusoidal_absolute",
            "relative_position",
            "alibi",
            "other_variant",
            "none_or_implicit",
        ):
            ocr_evidence_lines.extend(family_evidence.get(fam, [])[:1])
            if len(ocr_evidence_lines) >= 3:
                break

        notes = []
        if fallback_used:
            notes.append("classified_with_fallback_pass")

        out_rows.append(
            {
                "paper_dir": rel_dir,
                "dimension_label": row["final_label"],
                "pe_label": pe_label,
                "pe_components": ";".join(components),
                "stage_scope": stage_scope,
                "confidence": confidence,
                "ocr_evidence_lines": " | ".join(ocr_evidence_lines),
                "taskfile_evidence_lines": " | ".join(task_hints),
                "notes": ";".join(notes),
            }
        )

    candidate_ocr_txt.write_text("\n".join(sorted(candidate_ocr_lines)) + "\n", encoding="utf-8")
    write_tsv(hits_raw_tsv, raw_hits_rows)
    write_tsv(hit_windows_tsv, window_rows)

    out_rows.sort(key=lambda r: r["paper_dir"].lower())
    write_csv(
        out_csv,
        out_rows,
        [
            "paper_dir",
            "dimension_label",
            "pe_label",
            "pe_components",
            "stage_scope",
            "confidence",
            "ocr_evidence_lines",
            "taskfile_evidence_lines",
            "notes",
        ],
    )

    cnt = Counter(r["pe_label"] for r in out_rows)
    by_dim = Counter((r["dimension_label"], r["pe_label"]) for r in out_rows)

    summary = [
        "# Positional Encoding Summary",
        "",
        f"2D+/multi-D transformer papers processed: {len(out_rows)}",
        "",
        "PE labels:",
    ]
    for k in sorted(cnt):
        summary.append(f"- {k}: {cnt[k]}")
    summary += [
        "",
        "Top dimension x PE combinations:",
    ]
    for (d, p), n in sorted(by_dim.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))[:20]:
        summary.append(f"- {d} x {p}: {n}")
    summary += [
        "",
        "Files generated:",
        "- positional_encoding_results.csv",
        "- pe_candidate_paper_dirs.txt",
        "- pe_candidate_ocr_files.txt",
        "- pe_hits_raw.tsv",
        "- pe_hit_windows.tsv",
    ]
    summary_md.write_text("\n".join(summary) + "\n", encoding="utf-8")

    uncertain = [r for r in out_rows if r["pe_label"] == "unclear" or r["confidence"] == "low"]
    uncertain_lines = [
        "# Positional Encoding: Uncertain / Low Confidence",
        "",
        f"Count: {len(uncertain)}",
        "",
    ]
    for r in uncertain:
        uncertain_lines.append(
            f"- `{r['paper_dir']}` | pe_label={r['pe_label']} | confidence={r['confidence']} | evidence={r['ocr_evidence_lines']}"
        )
    uncertain_md.write_text("\n".join(uncertain_lines) + "\n", encoding="utf-8")


def main() -> None:
    paper_dirs = list_paper_dirs()
    run_step1_transformer_screen(paper_dirs)
    run_step2_task_dimensions()
    run_step3_positional_encoding()
    print("Pipeline completed.")
    print("Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
