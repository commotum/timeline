#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

DEFAULT_CSV="/Users/jake/Developer/timeline/ARXIV/arXiv.csv"
DEFAULT_OUT_DIR="/Users/jake/Developer/timeline/ARXIV"

CSV_PATH="${CSV_PATH:-$DEFAULT_CSV}"
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  CSV_PATH="$DEFAULT_CSV"
  OUT_DIR="$DEFAULT_OUT_DIR"
fi

csv_path="${1:-$CSV_PATH}"
out_dir="${2:-$OUT_DIR}"

LOG_FILE="${LOG_FILE:-$out_dir/download_arxiv.log}"
QUIET="${QUIET:-1}"
OVERWRITE="${OVERWRITE:-0}"
MAX_FILES="${MAX_FILES:-0}"

SLEEP_MIN="${SLEEP_MIN:-2}"
SLEEP_MAX="${SLEEP_MAX:-6}"
RETRIES="${RETRIES:-3}"

curl_user_agent="${CURL_USER_AGENT:-Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36}"
curl_alt_user_agent="${CURL_ALT_USER_AGENT:-Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15}"
curl_referer_mode="${CURL_REFERER_MODE:-origin}"
curl_referer="${CURL_REFERER:-}"
allow_insecure="${ALLOW_INSECURE:-0}"

curl_retries="${CURL_RETRIES:-2}"
curl_retry_delay="${CURL_RETRY_DELAY:-1}"
curl_connect_timeout="${CURL_CONNECT_TIMEOUT:-20}"
curl_max_time="${CURL_MAX_TIME:-300}"
curl_speed_limit="${CURL_SPEED_LIMIT:-1024}"
curl_speed_time="${CURL_SPEED_TIME:-30}"
curl_extra_flags="${CURL_EXTRA_FLAGS:-}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [CSV_PATH] [OUT_DIR]

Downloads/renames entries from an arXiv CSV into ARXIV.
- Local paths are renamed to the paper title.
- Remote URLs are downloaded with retries, backoff, and fallbacks.

Defaults:
  CSV_PATH: $DEFAULT_CSV
  OUT_DIR:  $DEFAULT_OUT_DIR

Env vars:
  LOG_FILE, QUIET, OVERWRITE, MAX_FILES, SLEEP_MIN, SLEEP_MAX, RETRIES,
  ALLOW_INSECURE, CURL_USER_AGENT, CURL_ALT_USER_AGENT,
  CURL_REFERER_MODE (origin|none|custom), CURL_REFERER,
  CURL_RETRIES, CURL_RETRY_DELAY, CURL_CONNECT_TIMEOUT, CURL_MAX_TIME,
  CURL_SPEED_LIMIT, CURL_SPEED_TIME, CURL_EXTRA_FLAGS
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

command -v curl >/dev/null 2>&1 || {
  echo "Error: curl is required but not found in PATH" >&2
  exit 1
}
command -v python3 >/dev/null 2>&1 || {
  echo "Error: python3 is required but not found in PATH" >&2
  exit 1
}

mkdir -p "$out_dir"
if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
fi

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  local level="$1"
  shift
  local line="[$(timestamp)] [$level] $*"
  printf '%s\n' "$line" >&2
  if [[ -n "$LOG_FILE" ]]; then
    printf '%s\n' "$line" >> "$LOG_FILE"
  fi
}

log_info() {
  log INFO "$@"
}

log_warn() {
  log WARN "$@"
}

log_error() {
  log ERROR "$@"
}

if [[ ! "$MAX_FILES" =~ ^[0-9]+$ ]]; then
  log_warn "Invalid MAX_FILES=$MAX_FILES; defaulting to 0"
  MAX_FILES=0
fi
if [[ ! "$RETRIES" =~ ^[0-9]+$ || "$RETRIES" == "0" ]]; then
  log_warn "Invalid RETRIES=$RETRIES; defaulting to 3"
  RETRIES=3
fi
if [[ ! "$SLEEP_MIN" =~ ^[0-9]+$ ]]; then
  log_warn "Invalid SLEEP_MIN=$SLEEP_MIN; defaulting to 2"
  SLEEP_MIN=2
fi
if [[ ! "$SLEEP_MAX" =~ ^[0-9]+$ ]]; then
  log_warn "Invalid SLEEP_MAX=$SLEEP_MAX; defaulting to 6"
  SLEEP_MAX=6
fi

rand_sleep() {
  local min_ms=$(( SLEEP_MIN * 1000 ))
  local max_ms=$(( SLEEP_MAX * 1000 ))
  if (( max_ms < min_ms )); then
    local tmp=$min_ms
    min_ms=$max_ms
    max_ms=$tmp
  fi
  if (( max_ms == min_ms )); then
    printf '%d.%03d' $((min_ms / 1000)) $((min_ms % 1000))
    return
  fi
  local span=$(( max_ms - min_ms + 1 ))
  local delay_ms=$(( RANDOM % span + min_ms ))
  printf '%d.%03d' $((delay_ms / 1000)) $((delay_ms % 1000))
}

cleanup_tmp() {
  if [[ ! -d "$out_dir" ]]; then
    return
  fi
  local files=()
  shopt -s nullglob
  files=("$out_dir"/.arxiv.*.tmp)
  shopt -u nullglob
  if (( ${#files[@]} > 0 )); then
    rm -f -- "${files[@]}" || true
  fi
}
cleanup_seen() {
  if [[ -n "${seen_file:-}" && -f "${seen_file}" ]]; then
    rm -f -- "${seen_file}" || true
  fi
}

cleanup_all() {
  cleanup_tmp
  cleanup_seen
}
trap cleanup_all EXIT

origin_from_url() {
  local url="$1"
  printf '%s' "$url" | sed -nE 's#^(https?://[^/]+).*#\1#p'
}

build_referer() {
  local url="$1"
  case "$curl_referer_mode" in
    none)
      printf ''
      ;;
    custom)
      printf '%s' "$curl_referer"
      ;;
    origin|*)
      origin_from_url "$url"
      ;;
  esac
}

format_curl_reason() {
  local exit_code="$1"
  local http_code="$2"
  if [[ "$exit_code" -eq 0 ]]; then
    printf 'http %s' "$http_code"
    return
  fi
  case "$exit_code" in
    6)
      printf 'curl: could not resolve host'
      ;;
    7)
      printf 'curl: failed to connect'
      ;;
    22)
      printf 'http %s' "$http_code"
      ;;
    28)
      printf 'curl: timeout'
      ;;
    35)
      printf 'curl: ssl connect error'
      ;;
    51|60)
      printf 'curl: ssl certificate problem'
      ;;
    52)
      printf 'curl: empty reply'
      ;;
    *)
      printf 'curl exit %s' "$exit_code"
      ;;
  esac
}

curl_last_code=""
curl_last_exit=0

curl_attempt() {
  local url="$1"
  local tmp="$2"
  local user_agent="$3"
  local referer="$4"
  local insecure="$5"
  local -a args=(
    -L
    --connect-timeout "$curl_connect_timeout"
    --max-time "$curl_max_time"
    --retry "$curl_retries"
    --retry-delay "$curl_retry_delay"
    --compressed
    -o "$tmp"
    -w "%{http_code}"
    -H "Accept: application/pdf,application/octet-stream;q=0.9,*/*;q=0.8"
    -H "Accept-Language: en-US,en;q=0.9"
    -A "$user_agent"
  )
  if [[ "$curl_speed_limit" != "0" && "$curl_speed_time" != "0" ]]; then
    args+=(--speed-time "$curl_speed_time" --speed-limit "$curl_speed_limit")
  fi
  [[ "$QUIET" == "1" ]] && args+=(-sS)
  [[ -n "$referer" ]] && args+=(-H "Referer: $referer")
  [[ "$insecure" == "1" ]] && args+=(-k)
  if [[ -n "$curl_extra_flags" ]]; then
    # shellcheck disable=SC2206
    args+=($curl_extra_flags)
  fi

  set +e
  curl_last_code="$(curl "${args[@]}" "$url")"
  curl_last_exit=$?
  set -e

  if [[ -z "$curl_last_code" ]]; then
    curl_last_code="000"
  fi
}

curl_success() {
  local exit_code="$1"
  local http_code="$2"
  local tmp="$3"
  [[ "$exit_code" -eq 0 && "$http_code" =~ ^2 && -s "$tmp" ]]
}

download_reason=""

download_with_fallbacks() {
  local url="$1"
  local dest="$2"
  local label="$3"
  local referer=""
  local exit_code=0
  local http_code="000"
  local reason=""
  local attempt_num=0

  referer="$(build_referer "$url")"

  attempt_request() {
    local ua="$1"
    local ref="$2"
    local insecure="$3"
    local tmp=""
    tmp="$(mktemp "${out_dir}/.arxiv.${label}.XXXXXX.tmp")"
    curl_attempt "$url" "$tmp" "$ua" "$ref" "$insecure"
    exit_code=$curl_last_exit
    http_code=$curl_last_code
    if curl_success "$exit_code" "$http_code" "$tmp"; then
      mv -f -- "$tmp" "$dest"
      return 0
    fi
    if [[ "$exit_code" -eq 0 && "$http_code" =~ ^2 ]]; then
      reason="empty response"
    else
      reason="$(format_curl_reason "$exit_code" "$http_code")"
    fi
    rm -f -- "$tmp" || true
    return 1
  }

  while (( attempt_num < RETRIES )); do
    attempt_num=$((attempt_num + 1))
    log_info "GET $url (attempt ${attempt_num}/${RETRIES})"

    if attempt_request "$curl_user_agent" "$referer" "0"; then
      return 0
    fi

    if [[ "$http_code" == "403" || "$http_code" == "429" ]]; then
      log_warn "HTTP $http_code for $label; retrying with alternate headers"
      if attempt_request "$curl_alt_user_agent" "$referer" "0"; then
        return 0
      fi
    fi

    if [[ "$exit_code" -eq 35 || "$exit_code" -eq 51 || "$exit_code" -eq 60 ]]; then
      if [[ "$allow_insecure" == "1" ]]; then
        log_warn "SSL verification failed for $label; retrying with --insecure"
        if attempt_request "$curl_user_agent" "$referer" "1"; then
          return 0
        fi
      fi
    fi

    if (( attempt_num < RETRIES )); then
      local backoff=$(( 2 ** attempt_num ))
      local jitter
      jitter="$(rand_sleep)"
      log_info "Retrying after ${backoff}s + ${jitter}s"
      sleep "$backoff"
      sleep "$jitter"
    fi
  done

  download_reason="$reason"
  return 1
}

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
    required = {"title", "url"}
    fields = set(reader.fieldnames or [])
    missing = sorted(required - fields)
    if missing:
        print(f"CSV missing required columns: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)
    for row in reader:
        title = (row.get("title") or "").strip()
        year = (row.get("year") or "").strip()
        url = (row.get("url") or "").strip()
        if not (title or year or url):
            continue
        def clean(value: str) -> str:
            return value.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        print(clean(title), clean(year), clean(url), sep="\t")
PY
}

if [[ ! -f "$csv_path" ]]; then
  log_error "CSV not found: $csv_path"
  exit 1
fi

sanitize_filename() {
  local name="$1"
  name="${name//\\/-}"
  name="${name//\//-}"
  name="${name//:/-}"
  name="${name//\*/-}"
  name="${name//\?/-}"
  name="${name//\"/-}"
  name="${name//</-}"
  name="${name//>/-}"
  name="${name//|/-}"
  name="$(printf '%s' "$name" | tr -s ' ' | sed -E 's/^ +| +$//g')"
  printf '%s' "$name"
}

declare -a renamed=()
declare -a downloaded=()
declare -a failed=()
declare -a skipped=()
line_num=0
processed=0
seen_file=""

log_info "Starting arXiv download: $csv_path -> $out_dir"
log_info "User-Agent: $curl_user_agent"

seen_file="$(mktemp "${out_dir}/.arxiv.seen.XXXXXX.tmp")"

while IFS=$'\t' read -r title year url; do
  line_num=$((line_num + 1))
  processed=$((processed + 1))
  if (( MAX_FILES > 0 && processed > MAX_FILES )); then
    log_info "Reached MAX_FILES=$MAX_FILES; stopping"
    break
  fi

  if [[ -z "$title" ]]; then
    failed+=("line $line_num | missing title")
    log_warn "Skipping line $line_num: missing title"
    continue
  fi
  if [[ -z "$url" ]]; then
    failed+=("line $line_num | missing url")
    log_warn "Skipping line $line_num: missing url"
    continue
  fi

  name="$(sanitize_filename "$title")"
  if [[ -z "$name" ]]; then
    failed+=("line $line_num | empty filename from title")
    log_warn "Skipping line $line_num: empty filename from title"
    continue
  fi

  if grep -Fqx -- "$name" "$seen_file"; then
    skipped+=("$name")
    log_warn "Duplicate entry for $name; skipping"
    continue
  fi
  printf '%s\n' "$name" >> "$seen_file"

  if [[ "$url" =~ ^https?:// ]]; then
    ext="$(get_ext_from_path "$url")"
    if [[ -z "$ext" ]]; then
      ext=".pdf"
    fi

    dest="${out_dir}/${name}${ext}"
    if [[ -s "$dest" && "$OVERWRITE" != "1" ]]; then
      skipped+=("$name")
      log_info "Skipping existing file: $dest"
      continue
    fi

    if download_with_fallbacks "$url" "$dest" "$name"; then
      downloaded+=("$name")
      log_info "Downloaded: $name"
      delay="$(rand_sleep)"
      log_info "Sleeping ${delay}s before next request"
      sleep "$delay"
    else
      failed+=("$name | $download_reason | $url")
      log_warn "Failed download: $name ($download_reason)"
    fi
  else
    if [[ "$url" == file://* ]]; then
      url="${url#file://}"
    fi
    if [[ "$url" != /* ]]; then
      url="$(dirname "$csv_path")/$url"
    fi
    ext="$(get_ext_from_path "$url")"
    if [[ -z "$ext" ]]; then
      ext=".pdf"
    fi

    dest="${out_dir}/${name}${ext}"
    if [[ ! -e "$url" ]]; then
      if [[ -e "$dest" ]]; then
        skipped+=("$name")
        log_info "Already renamed: $dest"
        continue
      fi
      failed+=("$name | missing local file: $url")
      log_warn "Missing local file: $url"
      continue
    fi
    if [[ "$url" == "$dest" ]]; then
      skipped+=("$name")
      log_info "Already named: $dest"
      continue
    fi
    if [[ -e "$dest" && "$OVERWRITE" != "1" ]]; then
      failed+=("$name | target exists: $dest")
      log_warn "Target exists, skipping: $dest"
      continue
    fi

    mv -f -- "$url" "$dest"
    renamed+=("$name")
    log_info "Renamed: $name"
  fi
done < <(csv_rows)

log_info "Done. Files saved in: $out_dir"
log_info "Report:"
log_info "Renamed local files (${#renamed[@]}):"
if ((${#renamed[@]})); then
  for item in "${renamed[@]}"; do
    log_info "  - $item"
  done
else
  log_info "  - none"
fi
log_info "Downloaded files (${#downloaded[@]}):"
if ((${#downloaded[@]})); then
  for item in "${downloaded[@]}"; do
    log_info "  - $item"
  done
else
  log_info "  - none"
fi
if ((${#failed[@]})); then
  log_info "Failed entries (${#failed[@]}):"
  for item in "${failed[@]}"; do
    log_info "  - $item"
  done
else
  log_info "Failed entries (0)"
fi
if ((${#skipped[@]})); then
  log_info "Skipped entries (${#skipped[@]}):"
  for item in "${skipped[@]}"; do
    log_info "  - $item"
  done
else
  log_info "Skipped entries (0)"
fi
