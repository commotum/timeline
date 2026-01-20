#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path


NEW_DOWNLOADS = Path("/home/jake/Developer/timeline/NEW/Downloads")
PDF2TXT_ROOT = Path("/home/jake/Developer/pdf2txt")
PDF_IN = PDF2TXT_ROOT / "pdf_in"
MD_OUT = PDF2TXT_ROOT / "md_out"
FIN_DIR = PDF2TXT_ROOT / "fin"
NEW_MARKDOWN = Path("/home/jake/Developer/timeline/NEW/Markdown")


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


def move_pdfs(src_dir: Path, dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for path in src_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".pdf":
            target = unique_destination(dest_dir / path.name)
            shutil.move(str(path), str(target))
            moved += 1
    return moved


def run_cmd(cmd, cwd: Path, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
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
    if not PDF2TXT_ROOT.exists():
        print(f"pdf2txt dir not found: {PDF2TXT_ROOT}", file=sys.stderr)
        return 1

    moved_pdfs = move_pdfs(NEW_DOWNLOADS, PDF_IN)
    print(f"Moved {moved_pdfs} PDF(s) to {PDF_IN}")

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
