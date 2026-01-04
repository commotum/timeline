#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from collections import defaultdict


PREFIX_BY_CLASS = {
    "POS-ENCDR": "01",
    "XFORM-DIM": "02",
    "COMP-REAS": "03",
    "DATA-BNCH": "04",
    "ML-FNDTNS": "05",
    "X-CONTEXT": "06",
    "MIS-CLASS": "07",
}


def sanitize_filename(name: str) -> str:
    name = name.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, "-")
    name = " ".join(name.split())
    return name


def build_class_folders(root: str) -> dict:
    folders = {}
    try:
        entries = os.listdir(root)
    except FileNotFoundError:
        raise SystemExit(f"Root directory not found: {root}")

    for entry in entries:
        full_path = os.path.join(root, entry)
        if not os.path.isdir(full_path):
            continue
        for class_code, prefix in PREFIX_BY_CLASS.items():
            if entry.startswith(f"{prefix} -"):
                folders[class_code] = entry

    missing = [code for code in PREFIX_BY_CLASS if code not in folders]
    if missing:
        raise SystemExit(f"Missing class folders for: {', '.join(missing)}")
    return folders


def candidate_bases(base: str, row_id: str):
    yield base
    if row_id:
        yield sanitize_filename(f"{base} - {row_id}")
        suffix = 2
        while True:
            yield sanitize_filename(f"{base} - {row_id}-{suffix}")
            suffix += 1
    else:
        suffix = 2
        while True:
            yield sanitize_filename(f"{base}-{suffix}")
            suffix += 1


def load_csv(path: str):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "year", "name", "title", "class"}
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise SystemExit(f"CSV missing required columns: {', '.join(missing)}")
        rows = list(reader)
        return rows, reader.fieldnames


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename BIBLIOTHEQUE PDFs to sanitized titles from BIBLIOTHEQUE.csv."
    )
    parser.add_argument(
        "--csv",
        default=os.path.join("BIBLIOTHEQUE", "BIBLIOTHEQUE.csv"),
        help="Path to BIBLIOTHEQUE.csv",
    )
    parser.add_argument(
        "--root",
        default="BIBLIOTHEQUE",
        help="Root BIBLIOTHEQUE directory containing class folders",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply renames and write updated CSV (default: dry-run)",
    )
    parser.add_argument(
        "--rename-md",
        action="store_true",
        help="Also rename matching .md sidecars when present",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file actions",
    )
    args = parser.parse_args()

    rows, fieldnames = load_csv(args.csv)
    class_folders = build_class_folders(args.root)

    existing_pdfs = {}
    for class_code, folder_name in class_folders.items():
        folder_path = os.path.join(args.root, folder_name)
        existing = {
            name
            for name in os.listdir(folder_path)
            if name.lower().endswith(".pdf") and os.path.isfile(os.path.join(folder_path, name))
        }
        existing_pdfs[class_code] = existing

    assigned_names = defaultdict(set)

    renamed = 0
    already_named = 0
    missing = 0
    skipped = 0
    md_renamed = 0
    md_conflict = 0

    for index, row in enumerate(rows, start=1):
        class_code = (row.get("class") or "").strip()
        if class_code not in class_folders:
            skipped += 1
            continue

        title = (row.get("title") or "").strip()
        if not title:
            missing += 1
            continue

        base = sanitize_filename(title)
        if not base:
            missing += 1
            continue

        row_id = (row.get("id") or "").strip()
        if not row_id:
            row_id = f"row-{index}"

        folder_name = class_folders[class_code]
        folder_path = os.path.join(args.root, folder_name)

        src_name = (row.get("name") or "").strip()
        if not src_name.lower().endswith(".pdf"):
            skipped += 1
            continue

        src_path = os.path.join(folder_path, src_name)
        src_exists = os.path.exists(src_path)

        chosen_name = None
        chosen_path = None

        for candidate_base in candidate_bases(base, row_id):
            candidate_name = f"{candidate_base}.pdf"
            if candidate_name in assigned_names[class_code]:
                continue

            candidate_path = os.path.join(folder_path, candidate_name)
            candidate_exists = candidate_name in existing_pdfs[class_code]

            if candidate_exists and src_exists and os.path.basename(src_path) != candidate_name:
                continue
            if not candidate_exists and not src_exists:
                continue

            chosen_name = candidate_name
            chosen_path = candidate_path
            break

        if not chosen_name:
            missing += 1
            continue

        assigned_names[class_code].add(chosen_name)

        if src_exists and os.path.basename(src_path) != chosen_name:
            if args.apply:
                os.rename(src_path, chosen_path)
            if args.rename_md:
                src_md = os.path.splitext(src_path)[0] + ".md"
                dest_md = os.path.splitext(chosen_path)[0] + ".md"
                if os.path.exists(src_md) and src_md != dest_md:
                    if os.path.exists(dest_md):
                        md_conflict += 1
                    elif args.apply:
                        os.rename(src_md, dest_md)
                        md_renamed += 1
            existing_pdfs[class_code].discard(os.path.basename(src_path))
            existing_pdfs[class_code].add(chosen_name)
            renamed += 1
            if args.verbose:
                print(f"renamed: {src_path} -> {chosen_path}")
        else:
            already_named += 1
            if args.verbose:
                print(f"kept: {chosen_path or src_path}")

        row["name"] = chosen_name

        if args.rename_md:
            # If PDFs are already renamed, fall back to legacy id-year .md naming.
            legacy_base = sanitize_filename(f"{row_id}-{row.get('year', '').strip()}")
            legacy_md = os.path.join(folder_path, f"{legacy_base}.md")
            dest_md = os.path.splitext(os.path.join(folder_path, chosen_name))[0] + ".md"
            if os.path.exists(legacy_md) and legacy_md != dest_md:
                if os.path.exists(dest_md):
                    md_conflict += 1
                elif args.apply:
                    os.rename(legacy_md, dest_md)
                    md_renamed += 1

    if args.apply:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("Done.")
    print(f"Renamed PDFs: {renamed}")
    print(f"Already named: {already_named}")
    print(f"Missing/Skipped: {missing + skipped}")
    if args.rename_md:
        print(f"Renamed .md sidecars: {md_renamed}")
        if md_conflict:
            print(f".md conflicts: {md_conflict}")
    if not args.apply:
        print("Dry-run only; re-run with --apply to make changes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
