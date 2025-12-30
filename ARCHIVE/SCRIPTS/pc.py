import argparse
import csv
import re
from pathlib import Path


ENTRY_LINE_RE = re.compile(r"^- \*\*(.+?)\*\*")
REF_DEF_RE = re.compile(r"^\[([0-9]+-[0-9]+)\]:\s*(.*)$")
REF_ID_RE = re.compile(r"\[([0-9]+-[0-9]+)\]")


def parse_title_year(line):
    match = ENTRY_LINE_RE.match(line)
    if not match:
        return "", ""
    title = match.group(1).strip()
    rest = line[match.end():]
    year = ""
    year_match = re.search(r"\(([^)]+)\)", rest)
    if year_match:
        year = year_match.group(1).strip()
    return title, year


def load_reference_map(lines):
    refs = {}
    for line in lines:
        match = REF_DEF_RE.match(line.strip())
        if match:
            refs[match.group(1)] = match.group(2).strip()
    return refs


def iter_entry_blocks(lines):
    starts = [idx for idx, line in enumerate(lines) if line.startswith("- **")]
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        yield lines[start:end]


def first_ref_id(lines):
    for line in lines:
        if REF_DEF_RE.match(line.strip()):
            continue
        match = REF_ID_RE.search(line)
        if match:
            return match.group(1)
    return ""


def build_rows(lines, refs):
    rows = []
    missing_refs = 0
    for block in iter_entry_blocks(lines):
        title, year = parse_title_year(block[0])
        ref_id = first_ref_id(block)
        url = refs.get(ref_id, "")
        if ref_id and not url:
            missing_refs += 1
        rows.append((year, title, url))
    return rows, missing_refs


def main():
    parser = argparse.ArgumentParser(
        description="Parse CONSOLIDATED.md into a CSV with year, title, and URL.",
    )
    parser.add_argument(
        "--input",
        default="CONSOLIDATED.md",
        help="Path to CONSOLIDATED.md",
    )
    parser.add_argument(
        "--output",
        default="CONSOLIDATED.csv",
        help="Path to output CSV",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    lines = input_path.read_text(encoding="utf-8").splitlines()
    refs = load_reference_map(lines)
    rows, missing_refs = build_rows(lines, refs)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", "title", "url"])
        writer.writerows(rows)

    print(f"wrote_rows={len(rows)} missing_ref_urls={missing_refs} output={output_path}")


if __name__ == "__main__":
    main()
