#!/usr/bin/env python3
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
NEW_ROOT = REPO_ROOT / "NEW"
NEW_DOWNLOADS = NEW_ROOT / "Downloads"
NEW_MARKDOWN = NEW_ROOT / "Markdown"
NEW_CSV = NEW_ROOT / "NEW.csv"

PDF2TXT_ROOT = Path(os.environ.get("PDF2TXT_ROOT", "/home/jake/Developer/pdf2txt"))
PDF_IN = PDF2TXT_ROOT / "pdf_in"
MD_OUT = PDF2TXT_ROOT / "md_out"
FIN_DIR = PDF2TXT_ROOT / "fin"


def unique_destination(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def sanitize_title(value):
    sanitized = value
    for ch in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]:
        sanitized = sanitized.replace(ch, "-")
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def load_expected_basenames(csv_path: Path):
    if not csv_path.exists():
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "title" not in fieldnames:
            return set()
        basenames = set()
        for row in reader:
            title = (row.get("title") or "").strip()
            if not title:
                continue
            basenames.add(sanitize_title(title))
    return basenames


def move_pdfs(src_dir: Path, dest_dir: Path, allowed_basenames) -> tuple[int, list]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    skipped = []
    for path in src_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() != ".pdf":
            continue
        if allowed_basenames and path.stem not in allowed_basenames:
            skipped.append(path.name)
            continue
        target = unique_destination(dest_dir / path.name)
        shutil.move(str(path), str(target))
        moved += 1
    return moved, skipped


def run_cmd(cmd, cwd: Path, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.pop("VIRTUAL_ENV", None)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def move_subfolders(src_dir: Path, dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    if not src_dir.exists():
        return moved
    for path in src_dir.iterdir():
        if path.is_dir():
            target = unique_destination(dest_dir / path.name)
            shutil.move(str(path), str(target))
            moved += 1
    return moved


def main() -> int:
    if not NEW_DOWNLOADS.exists():
        print(f"Downloads dir not found: {NEW_DOWNLOADS}", file=sys.stderr)
        return 1
    if not NEW_CSV.exists():
        print(f"NEW.csv not found: {NEW_CSV}", file=sys.stderr)
        return 1
    if not PDF2TXT_ROOT.exists():
        print(f"pdf2txt dir not found: {PDF2TXT_ROOT}", file=sys.stderr)
        return 1

    allowed_basenames = load_expected_basenames(NEW_CSV)
    if not allowed_basenames:
        print("No NEW.csv entries found; skipping extract.")
        return 0
    moved_pdfs, skipped = move_pdfs(NEW_DOWNLOADS, PDF_IN, allowed_basenames)
    print(f"Moved {moved_pdfs} PDF(s) to {PDF_IN}")
    if skipped:
        print("Skipped PDFs not present in NEW.csv:")
        for name in skipped:
            print(f"  - {name}")

    run_cmd(
        [
            "uv",
            "run",
            "marker",
            str(PDF_IN),
            "--force_ocr",
            "--workers",
            "4",
            "--output_dir",
            str(MD_OUT),
        ],
        cwd=PDF2TXT_ROOT,
        extra_env={"TORCH_DEVICE": "cuda"},
    )

    run_cmd([sys.executable, "fin.py"], cwd=PDF2TXT_ROOT)

    moved_dirs = move_subfolders(FIN_DIR, NEW_MARKDOWN)
    print(f"Moved {moved_dirs} folder(s) to {NEW_MARKDOWN}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
