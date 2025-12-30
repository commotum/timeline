#!/usr/bin/env bash
set -euo pipefail

csv_path="${1:-/home/jake/Developer/timeline/BIBLIOTHEQUE.csv}"
out_dir="${2:-/home/jake/Developer/timeline/BIBLIOTHEQUE}"

if [[ ! -f "$csv_path" ]]; then
  echo "CSV not found: $csv_path" >&2
  exit 1
fi

mkdir -p "$out_dir"

declare -a renamed=()
declare -a downloaded=()
declare -a failed=()
declare -a skipped=()
line_num=0

get_ext_from_path() {
  local path="$1"
  local clean="${path%%\?*}"
  clean="${clean%%\#*}"
  local base="${clean##*/}"
  if [[ "$base" == *.* ]]; then
    printf '.%s' "${base##*.}"
  fi
}

csv_rows() {
  python3 - "$csv_path" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        row_id = (row.get("id") or "").strip()
        year = (row.get("year") or "").strip()
        url = (row.get("url") or "").strip()
        if not (row_id or year or url):
            continue
        def clean(value: str) -> str:
            return value.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        print(clean(row_id), clean(year), clean(url), sep="\t")
PY
}

while IFS=$'\t' read -r row_id year url; do
  line_num=$((line_num + 1))

  if [[ -z "$row_id" || -z "$year" ]]; then
    failed+=("line $line_num | missing id/year")
    echo "Skipping line $line_num: missing id/year" >&2
    continue
  fi
  if [[ -z "$url" ]]; then
    failed+=("line $line_num | missing url")
    echo "Skipping line $line_num: missing url" >&2
    continue
  fi

  row_id=$(printf '%s' "$row_id" | tr '[:upper:]' '[:lower:]')
  name="${row_id}-${year}"

  if [[ "$url" =~ ^https?:// ]]; then
    ext="$(get_ext_from_path "$url")"
    if [[ -z "$ext" ]]; then
      ext=".pdf"
    fi

    dest="${out_dir}/${name}${ext}"
    if [[ -s "$dest" ]]; then
      skipped+=("$name")
      echo "Skipping existing file: $dest"
      continue
    fi

    tmp="$(mktemp "${out_dir}/.tmp.${name}.XXXXXX")"
    if curl -L --fail --retry 3 --retry-delay 1 -o "$tmp" "$url"; then
      mv "$tmp" "$dest"
      downloaded+=("$name")
      echo "Downloaded: $name"
    else
      rm -f "$tmp"
      failed+=("$name | $url")
      echo "Failed download: $name" >&2
    fi
  else
    if [[ "$url" == file://* ]]; then
      url="${url#file://}"
    fi
    if [[ "$url" != /* ]]; then
      url="$(dirname "$csv_path")/$url"
    fi
    if [[ ! -e "$url" ]]; then
      failed+=("$name | missing local file: $url")
      echo "Missing local file: $url" >&2
      continue
    fi

    ext="$(get_ext_from_path "$url")"
    if [[ -z "$ext" ]]; then
      ext=".pdf"
    fi

    dest="${out_dir}/${name}${ext}"
    if [[ "$url" == "$dest" ]]; then
      skipped+=("$name")
      echo "Already named: $dest"
      continue
    fi
    if [[ -e "$dest" ]]; then
      failed+=("$name | target exists: $dest")
      echo "Target exists, skipping: $dest" >&2
      continue
    fi

    mv "$url" "$dest"
    renamed+=("$name")
    echo "Renamed: $name"
  fi
done < <(csv_rows)

echo "Done. Files saved in: $out_dir"
echo "Report:"
echo "Renamed local files (${#renamed[@]}):"
if ((${#renamed[@]})); then
  printf '  - %s\n' "${renamed[@]}"
else
  echo "  - none"
fi
echo "Downloaded files (${#downloaded[@]}):"
if ((${#downloaded[@]})); then
  printf '  - %s\n' "${downloaded[@]}"
else
  echo "  - none"
fi
if ((${#failed[@]})); then
  echo "Failed entries (${#failed[@]}):"
  printf '  - %s\n' "${failed[@]}"
else
  echo "Failed entries (0)"
fi
if ((${#skipped[@]})); then
  echo "Skipped entries (${#skipped[@]}):"
  printf '  - %s\n' "${skipped[@]}"
else
  echo "Skipped entries (0)"
fi
