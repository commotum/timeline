#!/usr/bin/env python3
import argparse
import csv
import re
from urllib.parse import urlparse
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


def write_csv_atomic(path, fieldnames, rows):
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temp_path.replace(path)


def strip_url_suffix(value):
    cleaned = value.strip()
    if "?" in cleaned:
        cleaned = cleaned.split("?", 1)[0]
    if "#" in cleaned:
        cleaned = cleaned.split("#", 1)[0]
    return cleaned


def has_pdf_extension(value):
    cleaned = strip_url_suffix(value)
    return cleaned.lower().endswith(".pdf")


def normalize_whitespace(value):
    return " ".join(value.split())


def sanitize_title(value):
    sanitized = value
    for ch in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]:
        sanitized = sanitized.replace(ch, "-")
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


ARXIV_ID_RE = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z\-]+)?/\d{7})(?:v\d+)?$"
)


def normalize_arxiv_url(value):
    original = (value or "").strip()
    if not original:
        return original, False, None
    candidate = original
    if re.match(r"^(www\.)?arxiv\.org/", candidate):
        candidate = "https://" + candidate
    parsed = urlparse(candidate)
    if parsed.netloc not in ("arxiv.org", "www.arxiv.org"):
        return original, False, None
    path = parsed.path or ""
    if not (path.startswith("/abs/") or path.startswith("/pdf/")):
        return original, False, f"unsupported arXiv path '{path}'"
    parts = path.split("/", 2)
    if len(parts) < 3 or not parts[2]:
        return original, False, f"missing arXiv id in path '{path}'"
    arxiv_id = parts[2].strip("/")
    if arxiv_id.endswith(".pdf"):
        arxiv_id = arxiv_id[:-4]
    if not ARXIV_ID_RE.match(arxiv_id):
        return original, False, f"invalid arXiv id '{arxiv_id}'"
    normalized = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    if normalized != candidate:
        return normalized, True, None
    if normalized != original:
        return normalized, True, None
    return original, False, None


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
    repo_root = script_dir.parents[1]
    new_root = repo_root / "NEW"
    bib_root = repo_root / "BIBLIOTHEQUE"

    new_csv = Path(args.new_csv) if args.new_csv else new_root / "NEW.csv"
    bib_csv = Path(args.bib_csv) if args.bib_csv else bib_root / "BIBLIOTHEQUE.csv"
    out_csv = Path(args.out) if args.out else new_root / "duplicates.csv"
    downloads_dir = new_root / "Downloads"
    markdown_dir = new_root / "Markdown"

    if not new_csv.exists():
        print(f"[precheck] NEW.csv not found: {new_csv}")
        return 1
    if not bib_csv.exists():
        print(f"[precheck] BIBLIOTHEQUE.csv not found: {bib_csv}")
        return 1

    try:
        with new_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            new_fields = reader.fieldnames or []
            if not new_fields:
                raise ValueError(f"NEW.csv missing header ({new_csv})")
            missing = [f for f in ("year", "title", "url") if f not in new_fields]
            if missing:
                raise ValueError(
                    f"NEW.csv missing columns: {', '.join(missing)} ({new_csv})"
                )
            new_rows = list(reader)
    except Exception as exc:
        print(f"[precheck] failed to read NEW.csv: {exc}")
        return 1

    missing_fields = []
    invalid_years = []
    invalid_urls = []
    arxiv_errors = []
    url_fixes = []
    title_fixes = []

    for idx, row in enumerate(new_rows, start=2):
        title_raw = (row.get("title") or "").strip()
        url_raw = (row.get("url") or "").strip()
        year_raw = (row.get("year") or "").strip()

        if not title_raw:
            missing_fields.append((idx, "title"))
        if not url_raw:
            missing_fields.append((idx, "url"))

        if year_raw and (not year_raw.isdigit() or len(year_raw) != 4):
            invalid_years.append((idx, title_raw, year_raw))

        title = normalize_whitespace(title_raw)
        if title != title_raw:
            row["title"] = title
            title_fixes.append((idx, title_raw, title))

        url = url_raw
        normalized_url, changed, error = normalize_arxiv_url(url)
        if error:
            arxiv_errors.append((idx, title, url, error))
            continue
        if changed:
            row["url"] = normalized_url
            url_fixes.append((idx, title, url, normalized_url))
            url = normalized_url

        if url and not has_pdf_extension(url):
            invalid_urls.append((idx, title, url))

    if missing_fields:
        print("[precheck] NEW.csv missing required fields:")
        for line_no, field in missing_fields:
            print(f"  - line {line_no}: missing {field}")
        return 1

    if invalid_years:
        print("[precheck] NEW.csv entries with invalid year (expected YYYY):")
        for line_no, title, year in invalid_years:
            print(f"  - line {line_no}: {title} | {year}")
        return 1

    if arxiv_errors:
        print("[precheck] NEW.csv entries with invalid arXiv URLs:")
        for line_no, title, url, error in arxiv_errors:
            print(f"  - line {line_no}: {title} | {url} | {error}")
        return 1

    if url_fixes:
        print("[precheck] normalized arXiv URLs:")
        for line_no, title, before, after in url_fixes:
            print(f"  - line {line_no}: {title} | {before} -> {after}")

    if title_fixes:
        print("[precheck] normalized titles (whitespace cleanup):")
        for line_no, before, after in title_fixes:
            print(f"  - line {line_no}: {before} -> {after}")

    if invalid_urls:
        print("[precheck] NEW.csv entries without .pdf URL extension:")
        for line_no, title, url in invalid_urls:
            print(f"  - line {line_no}: {title} | {url}")
        return 1

    dup_url = {}
    dup_title = {}
    dup_filename = {}
    for idx, row in enumerate(new_rows, start=2):
        title = (row.get("title") or "").strip()
        url = (row.get("url") or "").strip()
        if url:
            dup_url.setdefault(url, []).append(idx)
        if title:
            dup_title.setdefault(title, []).append(idx)
            filename = sanitize_title(title)
            if filename:
                dup_filename.setdefault(filename, []).append(idx)

    url_dupes = {k: v for k, v in dup_url.items() if len(v) > 1}
    title_dupes = {k: v for k, v in dup_title.items() if len(v) > 1}
    filename_dupes = {k: v for k, v in dup_filename.items() if len(v) > 1}

    if url_dupes:
        print("[precheck] NEW.csv contains duplicate URLs:")
        for url, lines in url_dupes.items():
            print(f"  - lines {', '.join(map(str, lines))}: {url}")
        return 1

    if title_dupes:
        print("[precheck] NEW.csv contains duplicate titles:")
        for title, lines in title_dupes.items():
            print(f"  - lines {', '.join(map(str, lines))}: {title}")
        return 1

    if filename_dupes:
        print("[precheck] NEW.csv contains filename collisions after sanitization:")
        for filename, lines in filename_dupes.items():
            print(f"  - lines {', '.join(map(str, lines))}: {filename}")
        return 1

    try:
        bib_rows = load_rows(bib_csv, ["year", "title", "url"])
    except Exception as exc:
        print(f"[precheck] failed to read BIBLIOTHEQUE.csv: {exc}")
        return 1

    by_url, by_title = index_bib(bib_rows)

    duplicates = []
    seen_pairs = set()
    duplicate_count = 0
    duplicate_new_indices = set()

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
            duplicate_new_indices.add(new_idx)
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

    csv_changed = bool(url_fixes or title_fixes)

    if downloads_dir.exists():
        expected = {sanitize_title((row.get("title") or "").strip()) for row in new_rows}
        ignore_files = {"download_new.log"}
        extra_files = []
        for item in downloads_dir.iterdir():
            if not item.is_file():
                continue
            if item.name.startswith(".") or item.name in ignore_files:
                continue
            if item.stem not in expected:
                extra_files.append(item.name)
        if extra_files:
            print("[precheck] warning: Downloads has files not in NEW.csv:")
            for name in sorted(extra_files):
                print(f"  - {name}")

    if markdown_dir.exists():
        expected = {sanitize_title((row.get("title") or "").strip()) for row in new_rows}
        extra_folders = []
        for item in markdown_dir.iterdir():
            if not item.is_dir():
                continue
            if item.name.startswith("."):
                continue
            if sanitize_title(item.name) not in expected:
                extra_folders.append(item.name)
        if extra_folders:
            print("[precheck] warning: Markdown has folders not in NEW.csv:")
            for name in sorted(extra_folders):
                print(f"  - {name}")

    if duplicates:
        write_duplicates(out_csv, duplicates)
        print(
            f"[precheck] found {duplicate_count} duplicate match(es); wrote {out_csv}"
        )
        remaining_rows = [
            row for idx, row in enumerate(new_rows) if idx not in duplicate_new_indices
        ]
        try:
            write_csv_atomic(new_csv, new_fields, remaining_rows)
            print(
                f"[precheck] removed {len(duplicate_new_indices)} duplicate NEW row(s) from {new_csv}"
            )
        except Exception as exc:
            print(f"[precheck] failed to update NEW.csv: {exc}")
            return 1
        csv_changed = True
    else:
        if out_csv.exists():
            out_csv.unlink()
            print(f"[precheck] no duplicates; removed stale {out_csv}")
        else:
            print("[precheck] no duplicates found")

    if csv_changed and not duplicates:
        try:
            write_csv_atomic(new_csv, new_fields, new_rows)
            print(f"[precheck] updated NEW.csv normalization fixes")
        except Exception as exc:
            print(f"[precheck] failed to update NEW.csv: {exc}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
