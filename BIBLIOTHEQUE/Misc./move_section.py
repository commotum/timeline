#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
import sys
import unicodedata


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    for ch in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        value = value.replace(ch, "-")
    for ch in ("\u2018", "\u2019", "\u02bc"):
        value = value.replace(ch, "'")
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def build_root_pdf_index(ocr_root: str) -> dict:
    pdf_by_norm = {}
    for name in os.listdir(ocr_root):
        path = os.path.join(ocr_root, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith(".pdf"):
            continue
        base = os.path.splitext(name)[0]
        key = norm(base)
        pdf_by_norm.setdefault(key, []).append(name)
    return pdf_by_norm


def move_section(section, sectioned_path, ocr_root, dry_run=False):
    with open(sectioned_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    pdf_by_norm = build_root_pdf_index(ocr_root)

    moved_dirs = 0
    moved_pdfs = 0
    missing_dirs = 0
    missing_pdfs = []
    conflicts_dirs = 0
    conflicts_pdfs = []
    ambiguous = []
    processed = 0

    for row in rows:
        if (row.get("section") or "").strip() != section:
            continue
        processed += 1
        file_stem = (row.get("filename") or "").strip()
        class_code = (row.get("class") or "").strip() or "UNKNOWN"
        class_id = (row.get("class_id") or "").strip()
        if not file_stem or not class_id:
            continue
        try:
            class_num = int(class_id)
        except ValueError:
            continue
        class_folder = f"{class_num:02d}_{class_code}"
        dest_bucket = os.path.join(ocr_root, class_folder)
        if not dry_run:
            os.makedirs(dest_bucket, exist_ok=True)

        src_dir = os.path.join(ocr_root, file_stem)
        dest_dir = os.path.join(dest_bucket, file_stem)

        if os.path.isdir(src_dir):
            if os.path.exists(dest_dir):
                conflicts_dirs += 1
            else:
                if not dry_run:
                    shutil.move(src_dir, dest_dir)
                moved_dirs += 1
        else:
            if not os.path.isdir(dest_dir):
                missing_dirs += 1

        direct_pdf = None
        for ext in (".pdf", ".PDF"):
            candidate = file_stem + ext
            if os.path.isfile(os.path.join(ocr_root, candidate)):
                direct_pdf = candidate
                break

        if direct_pdf:
            src_name = direct_pdf
        else:
            key = norm(file_stem)
            candidates = pdf_by_norm.get(key, [])
            if len(candidates) == 1:
                src_name = candidates[0]
            elif len(candidates) > 1:
                ambiguous.append((file_stem, candidates))
                continue
            else:
                missing_pdfs.append(file_stem)
                continue

        src_path = os.path.join(ocr_root, src_name)
        dest_path = os.path.join(dest_dir, src_name)

        if os.path.exists(dest_path):
            conflicts_pdfs.append((file_stem, src_name))
            continue

        if not dry_run:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.move(src_path, dest_path)
        moved_pdfs += 1

    return {
        "section": section,
        "processed": processed,
        "moved_dirs": moved_dirs,
        "moved_pdfs": moved_pdfs,
        "missing_dirs": missing_dirs,
        "missing_pdfs": missing_pdfs,
        "conflicts_dirs": conflicts_dirs,
        "conflicts_pdfs": conflicts_pdfs,
        "ambiguous": ambiguous,
        "dry_run": dry_run,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Move OCR folders and PDFs for a given section into class buckets."
    )
    parser.add_argument("--section", required=True, help="Section label, e.g. POS-ENCDR-01")
    parser.add_argument(
        "--sectioned",
        default=os.path.join("BIBLIOTHEQUE", "SECTIONED.csv"),
        help="Path to SECTIONED.csv",
    )
    parser.add_argument(
        "--ocr-root",
        default=os.path.join("BIBLIOTHEQUE", "OCR"),
        help="Root OCR folder",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not move files")
    args = parser.parse_args()

    result = move_section(args.section, args.sectioned, args.ocr_root, args.dry_run)
    print(f"Section: {result['section']}")
    print(f"Processed rows: {result['processed']}")
    print(f"Moved folders: {result['moved_dirs']}")
    print(f"Moved PDFs: {result['moved_pdfs']}")
    print(f"Missing folders: {result['missing_dirs']}")
    print(f"Missing PDFs: {len(result['missing_pdfs'])}")
    print(f"Conflicts (folders): {result['conflicts_dirs']}")
    print(f"Conflicts (pdfs): {len(result['conflicts_pdfs'])}")
    print(f"Ambiguous matches: {len(result['ambiguous'])}")
    if result["missing_pdfs"]:
        print("Missing PDFs sample: " + ", ".join(result["missing_pdfs"][:5]))
    if result["ambiguous"]:
        print("Ambiguous sample: " + result["ambiguous"][0][0])
    if result["dry_run"]:
        print("Dry-run only; no files moved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
