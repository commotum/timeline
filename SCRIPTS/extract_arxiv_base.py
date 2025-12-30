#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from urllib.parse import urlparse


def is_arxiv_base_url(value: str) -> bool:
    value = (value or "").strip()
    if not value:
        return False
    parsed = urlparse(value)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host:
        if host == "arxiv.org" and parsed.scheme in ("http", "https", ""):
            return True
        return False
    base = value.split("/")[0].lower()
    return base in ("arxiv.org", "www.arxiv.org")


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return None, []
        rows = list(reader)
    return header, rows


def write_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def read_output_header(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return None


def append_rows(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(header)
        writer.writerows(rows)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_chunked = repo_root / "CHUNKED"
    default_output = repo_root / "ARXIV" / "arXiv-base.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Move rows with arXiv base URLs from CHUNKED CSVs into arXiv-base.csv."
        )
    )
    parser.add_argument(
        "--chunked-dir",
        default=str(default_chunked),
        help="Directory containing the chunked CSV files.",
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
        help="Output CSV path for arXiv base URL rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report matches without modifying any files.",
    )
    args = parser.parse_args()

    chunked_dir = Path(args.chunked_dir)
    output_path = Path(args.output)

    csv_paths = sorted(chunked_dir.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSV files found in {chunked_dir}")

    output_header = read_output_header(output_path)
    total_matched = 0
    total_rows = 0

    for path in csv_paths:
        header, rows = read_csv(path)
        if header is None:
            continue

        url_idx = None
        for i, col in enumerate(header):
            if col.strip().lower() == "url":
                url_idx = i
                break
        if url_idx is None:
            url_idx = 2

        matched = []
        kept = []
        for row in rows:
            url = row[url_idx] if url_idx < len(row) else ""
            if is_arxiv_base_url(url):
                matched.append(row)
            else:
                kept.append(row)

        if matched:
            if output_header is None:
                output_header = header
            if header != output_header:
                raise SystemExit(
                    f"Header mismatch in {path}: {header} != {output_header}"
                )

            if not args.dry_run:
                append_rows(output_path, output_header, matched)
                write_csv(path, header, kept)

        total_rows += len(rows)
        total_matched += len(matched)
        print(f"{path.name}: moved {len(matched)} of {len(rows)} rows")

    print(f"Total moved: {total_matched} of {total_rows} rows")
    if args.dry_run:
        print("Dry run only; no files were modified.")


if __name__ == "__main__":
    main()
