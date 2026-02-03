#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def load_rows(path, required_fields):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [field for field in required_fields if field not in fieldnames]
        if missing:
            raise ValueError(f"CSV missing columns: {', '.join(missing)} ({path})")
        return list(reader)


def index_bib(rows):
    by_url = {}
    by_title = {}
    for idx, row in enumerate(rows):
        url = (row.get("url") or "").strip()
        title = (row.get("title") or "").strip()
        if url:
            by_url.setdefault(url, []).append(idx)
        if title:
            by_title.setdefault(title, []).append(idx)
    return by_url, by_title


def write_duplicates(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["year", "title", "source", "url"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precheck NEW.csv for duplicates already in BIBLIOTHEQUE.csv."
    )
    parser.add_argument("--new-csv", default=None, help="Path to NEW.csv")
    parser.add_argument("--bib-csv", default=None, help="Path to BIBLIOTHEQUE.csv")
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write duplicates.csv (default: NEW/duplicates.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    new_root = repo_root / "NEW"
    bib_root = repo_root / "BIBLIOTHEQUE"

    new_csv = Path(args.new_csv) if args.new_csv else new_root / "NEW.csv"
    bib_csv = Path(args.bib_csv) if args.bib_csv else bib_root / "BIBLIOTHEQUE.csv"
    out_csv = Path(args.out) if args.out else new_root / "duplicates.csv"

    if not new_csv.exists():
        print(f"[precheck] NEW.csv not found: {new_csv}")
        return 1
    if not bib_csv.exists():
        print(f"[precheck] BIBLIOTHEQUE.csv not found: {bib_csv}")
        return 1

    try:
        new_rows = load_rows(new_csv, ["year", "title", "url"])
        bib_rows = load_rows(bib_csv, ["year", "title", "url"])
    except Exception as exc:
        print(f"[precheck] failed to read CSVs: {exc}")
        return 1

    by_url, by_title = index_bib(bib_rows)

    duplicates = []
    seen_pairs = set()
    duplicate_count = 0

    for new_idx, new_row in enumerate(new_rows):
        new_url = (new_row.get("url") or "").strip()
        new_title = (new_row.get("title") or "").strip()
        matches = set()
        if new_url and new_url in by_url:
            matches.update(by_url[new_url])
        if new_title and new_title in by_title:
            matches.update(by_title[new_title])
        if not matches:
            continue
        for bib_idx in sorted(matches):
            pair_key = (new_idx, bib_idx)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            duplicates.append(
                {
                    "year": (new_row.get("year") or "").strip(),
                    "title": new_title,
                    "source": "NEW",
                    "url": new_url,
                }
            )
            bib_row = bib_rows[bib_idx]
            duplicates.append(
                {
                    "year": (bib_row.get("year") or "").strip(),
                    "title": (bib_row.get("title") or "").strip(),
                    "source": "BIBLIOTHEQUE",
                    "url": (bib_row.get("url") or "").strip(),
                }
            )
            duplicate_count += 1

    if duplicates:
        write_duplicates(out_csv, duplicates)
        print(
            f"[precheck] found {duplicate_count} duplicate match(es); wrote {out_csv}"
        )
    else:
        if out_csv.exists():
            out_csv.unlink()
            print(f"[precheck] no duplicates; removed stale {out_csv}")
        else:
            print("[precheck] no duplicates found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
