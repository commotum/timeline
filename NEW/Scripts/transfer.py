#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from pathlib import Path


ARROW_CHAR = "\u21a6"


def normalize_name(value):
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def sanitize_title(value):
    if value is None:
        return ""
    sanitized = value
    for ch in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]:
        sanitized = sanitized.replace(ch, "-")
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def read_class_mappings(path):
    class_id_to_name = {}
    name_to_code = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            match = re.match(r"^(\d+)\.\s+\*\*(.+?)\*\*", stripped)
            if match:
                class_id = int(match.group(1))
                class_name = match.group(2).strip()
                class_id_to_name[class_id] = class_name
                continue
            if ARROW_CHAR in stripped:
                parts = [part.strip() for part in stripped.split(ARROW_CHAR, 1)]
                if len(parts) == 2 and parts[0] and parts[1]:
                    name_to_code[normalize_name(parts[0])] = parts[1]

    class_id_to_code = {}
    missing = []
    for class_id in range(1, 8):
        class_name = class_id_to_name.get(class_id)
        if not class_name:
            missing.append(str(class_id))
            continue
        class_code = name_to_code.get(normalize_name(class_name))
        if not class_code:
            missing.append(str(class_id))
            continue
        class_id_to_code[class_id] = class_code

    if missing:
        raise ValueError(
            f"Missing class mappings for ids: {', '.join(missing)} in {path}"
        )
    return class_id_to_code


def find_class_folders(bib_root):
    code_to_folder = {}
    for entry in bib_root.iterdir():
        if not entry.is_dir():
            continue
        match = re.match(r"^\d{2}_(.+)$", entry.name)
        if not match:
            continue
        code = match.group(1)
        if code in code_to_folder:
            raise ValueError(f"Duplicate class folder for code {code}")
        code_to_folder[code] = entry
    return code_to_folder


def load_csv_rows(path, required_fields):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [field for field in required_fields if field not in fieldnames]
        if missing:
            raise ValueError(f"CSV missing columns: {', '.join(missing)} ({path})")
        rows = list(reader)
    return fieldnames, rows


def build_new_csv_index(rows):
    by_sanitized = {}
    by_normalized = {}
    for idx, row in enumerate(rows):
        title = (row.get("title") or "").strip()
        if not title:
            continue
        sanitized = sanitize_title(title)
        normalized = normalize_name(title)
        by_sanitized.setdefault(sanitized, []).append(idx)
        by_normalized.setdefault(normalized, []).append(idx)
    return by_sanitized, by_normalized


def find_row_index_for_folder(folder_name, by_sanitized, by_normalized):
    folder_key = sanitize_title(folder_name)
    matches = by_sanitized.get(folder_key)
    if matches:
        if len(matches) == 1:
            return matches[0], None
        return None, f"Multiple CSV rows match sanitized title '{folder_key}'"
    normalized = normalize_name(folder_name)
    matches = by_normalized.get(normalized)
    if matches:
        if len(matches) == 1:
            return matches[0], None
        return None, f"Multiple CSV rows match normalized title '{normalized}'"
    return None, f"No CSV row found for folder '{folder_name}'"


def find_classification(folder_path):
    class_files = []
    for entry in folder_path.iterdir():
        if not entry.is_file():
            continue
        match = re.match(r"^CLASS_([1-7])\.md$", entry.name)
        if match:
            class_files.append((entry, int(match.group(1))))
    if not class_files:
        return None, "Missing CLASS_<id>.md"
    if len(class_files) > 1:
        return None, "Multiple CLASS_<id>.md files found"
    return class_files[0][1], None


def rows_equivalent(existing, target):
    for key in ("year", "title", "class", "class_id", "basename", "url"):
        existing_value = (existing.get(key) or "").strip()
        target_value = (target.get(key) or "").strip()
        if existing_value != target_value:
            return False
    return True


def sort_bibliotheque_rows(rows):
    def sort_key(row):
        year_raw = (row.get("year") or "").strip()
        try:
            year_val = int(year_raw)
        except ValueError:
            year_val = 9999
        class_raw = (row.get("class_id") or "").strip()
        try:
            class_val = int(class_raw)
        except ValueError:
            class_val = 999
        title_val = (row.get("title") or "").strip().lower()
        return (year_val, class_val, title_val)

    return sorted(rows, key=sort_key)


def write_csv_atomic(path, fieldnames, rows):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def append_bibliotheque_rows(path, fieldnames, rows):
    if not rows:
        return
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transfer classified NEW/Markdown entries into BIBLIOTHEQUE."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing or moving files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    new_root = repo_root / "NEW"
    new_markdown = new_root / "Markdown"
    new_csv_path = new_root / "NEW.csv"
    bib_root = repo_root / "BIBLIOTHEQUE"
    bib_csv_path = bib_root / "BIBLIOTHEQUE.csv"
    class_map_path = bib_root / "00 - Class.md"

    if not new_markdown.is_dir():
        print(f"Markdown folder not found: {new_markdown}")
        return 1
    if not new_csv_path.is_file():
        print(f"NEW.csv not found: {new_csv_path}")
        return 1
    if not bib_csv_path.is_file():
        print(f"BIBLIOTHEQUE.csv not found: {bib_csv_path}")
        return 1
    if not class_map_path.is_file():
        print(f"Class map not found: {class_map_path}")
        return 1

    try:
        class_id_to_code = read_class_mappings(class_map_path)
        code_to_folder = find_class_folders(bib_root)
    except Exception as exc:
        print(f"Failed to load class mappings: {exc}")
        return 1

    class_id_to_folder = {}
    for class_id, code in class_id_to_code.items():
        folder = code_to_folder.get(code)
        if not folder:
            print(f"Missing class folder for code {code}")
            return 1
        class_id_to_folder[class_id] = folder

    try:
        new_fields, new_rows = load_csv_rows(new_csv_path, ["year", "title", "url"])
    except Exception as exc:
        print(f"Failed to load NEW.csv: {exc}")
        return 1

    by_sanitized, by_normalized = build_new_csv_index(new_rows)

    try:
        bib_fields, bib_rows = load_csv_rows(
            bib_csv_path,
            ["year", "title", "class", "class_id", "basename", "url"],
        )
    except Exception as exc:
        print(f"Failed to load BIBLIOTHEQUE.csv: {exc}")
        return 1

    existing_by_basename = {}
    for row in bib_rows:
        basename = (row.get("basename") or "").strip()
        if not basename:
            continue
        if basename not in existing_by_basename:
            existing_by_basename[basename] = row

    actions = []
    errors = []
    for entry in sorted(new_markdown.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue

        class_id, error = find_classification(entry)
        if error:
            errors.append(f"{entry.name}: {error}")
            continue

        class_code = class_id_to_code.get(class_id)
        if not class_code:
            errors.append(f"{entry.name}: Unknown class id {class_id}")
            continue

        row_index, error = find_row_index_for_folder(
            entry.name, by_sanitized, by_normalized
        )
        if error:
            errors.append(f"{entry.name}: {error}")
            continue

        row = new_rows[row_index]
        target_row = {
            "year": (row.get("year") or "").strip(),
            "title": (row.get("title") or "").strip(),
            "class": class_code,
            "class_id": str(class_id),
            "basename": entry.name,
            "url": (row.get("url") or "").strip(),
        }

        existing = existing_by_basename.get(entry.name)
        if existing and not rows_equivalent(existing, target_row):
            errors.append(
                f"{entry.name}: existing BIBLIOTHEQUE entry does not match NEW.csv"
            )
            continue

        dest_root = class_id_to_folder.get(class_id)
        dest_path = dest_root / entry.name
        if dest_path.exists():
            errors.append(f"{entry.name}: destination already exists ({dest_path})")
            continue

        actions.append(
            {
                "basename": entry.name,
                "source": entry,
                "dest": dest_path,
                "row_index": row_index,
                "bib_row": target_row,
                "already_in_bib": existing is not None,
            }
        )

    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")

    if not actions:
        print("No eligible entries to transfer.")
        return 1 if errors else 0

    if args.dry_run:
        add_count = sum(1 for action in actions if not action["already_in_bib"])
        print(f"Dry run: would add {add_count} entries to BIBLIOTHEQUE.csv")
        print(f"Dry run: would move {len(actions)} folders into BIBLIOTHEQUE")
        print(f"Dry run: would remove {len(actions)} rows from NEW.csv")
        return 1 if errors else 0

    rows_to_append = [
        action["bib_row"] for action in actions if not action["already_in_bib"]
    ]
    try:
        append_bibliotheque_rows(bib_csv_path, bib_fields, rows_to_append)
    except Exception as exc:
        print(f"Failed to append to BIBLIOTHEQUE.csv: {exc}")
        return 1

    moved_indices = set()
    for action in actions:
        try:
            shutil.move(str(action["source"]), str(action["dest"]))
            moved_indices.add(action["row_index"])
        except Exception as exc:
            print(f"Move failed for {action['basename']}: {exc}")

    if moved_indices:
        remaining_rows = [
            row for idx, row in enumerate(new_rows) if idx not in moved_indices
        ]
        try:
            write_csv_atomic(new_csv_path, new_fields, remaining_rows)
        except Exception as exc:
            print(f"Failed to update NEW.csv: {exc}")
            return 1

    try:
        updated_bib_rows = bib_rows + rows_to_append
        sorted_bib_rows = sort_bibliotheque_rows(updated_bib_rows)
        write_csv_atomic(bib_csv_path, bib_fields, sorted_bib_rows)
    except Exception as exc:
        print(f"Failed to sort BIBLIOTHEQUE.csv: {exc}")
        return 1

    print(f"Transferred {len(moved_indices)} folders into BIBLIOTHEQUE.")
    if moved_indices and len(moved_indices) != len(actions):
        print(
            "Some folders failed to move; rerun the script after resolving issues."
        )
        return 1

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
