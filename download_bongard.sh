#!/usr/bin/env bash

# Bongard Problem downloader
# - Downloads https://oebp.org/BP1 ... https://oebp.org/BP1290
# - Saves to /home/jake/Developer/arc-analysis/BONGARD/BPs/BP001.html ... BP1290.html
# - Sequential requests, randomized delay, retries with backoff, and logging

set -Eeuo pipefail
IFS=$'\n\t'

# Defaults (can be overridden via env vars before running)
BASE_URL="${BASE_URL:-https://oebp.org/BP}"
START="${START:-1}"
END="${END:-1290}"
OUT_DIR="${OUT_DIR:-/home/jake/Developer/arc-analysis/BONGARD/BPs}"
SLEEP_MIN="${SLEEP_MIN:-2}"       # seconds (integer)
SLEEP_MAX="${SLEEP_MAX:-6}"       # seconds (integer)
RETRIES="${RETRIES:-3}"
UA="${UA:-arc-analysis-bongard-downloader/1.0 (script: download_bongard.sh; purpose: research archival; url: https://oebp.org)}"
LOG_FILE="${LOG_FILE:-$OUT_DIR/download_bongard.log}"

usage() {
  cat <<USAGE
Usage: $(basename "$0")

Downloads Bongard Problem pages BP1..BP1290 to:
  $OUT_DIR

Notes:
- Configure via env vars if desired:
    BASE_URL, START, END, OUT_DIR, SLEEP_MIN, SLEEP_MAX, RETRIES, UA, LOG_FILE
- Requests are sequential with randomized sleep between them.
- Errors are retried up to \"$RETRIES\" times with backoff; failures are logged.

Examples:
  OUT_DIR="$OUT_DIR" $(basename "$0")
  START=10 END=25 $(basename "$0")
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

mkdir -p "$OUT_DIR"
# Ensure log directory exists (usually same as OUT_DIR)
mkdir -p "$(dirname "$LOG_FILE")"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] [$1] ${*:2}" | tee -a "$LOG_FILE" >&2; }
log_info() { log INFO "$@"; }
log_warn() { log WARN "$@"; }
log_error() { log ERROR "$@"; }

# Random sleep between SLEEP_MIN and SLEEP_MAX seconds, with millisecond jitter
rand_sleep() {
  local min_ms=$(( SLEEP_MIN * 1000 ))
  local max_ms=$(( SLEEP_MAX * 1000 ))
  if (( max_ms < min_ms )); then
    # swap if misconfigured
    local tmp=$min_ms; min_ms=$max_ms; max_ms=$tmp
  fi
  if (( max_ms == min_ms )); then
    printf '%d.%03d' $((min_ms/1000)) $((min_ms%1000))
    return
  fi
  local span=$(( max_ms - min_ms + 1 ))
  local delay_ms=$(( RANDOM % span + min_ms ))
  printf '%d.%03d' $((delay_ms/1000)) $((delay_ms%1000))
}

cleanup_tmp() {
  # Remove any leftover temp files from previous runs
  local files=()
  shopt -s nullglob
  files=("$OUT_DIR"/.BP*.tmp)
  shopt -u nullglob
  if (( ${#files[@]} > 0 )); then
    rm -f -- "${files[@]}" || true
  fi
}
trap cleanup_tmp EXIT

success_count=0
skip_existing_count=0
fail_count=0

log_info "Starting Bongard download: BP${START}..BP${END} -> $OUT_DIR"
log_info "User-Agent: $UA"

for (( i=START; i<=END; i++ )); do
  padded=$(printf '%03d' "$i")
  url="${BASE_URL}${i}"
  out_file="$OUT_DIR/BP${padded}.html"

  if [[ -s "$out_file" ]]; then
    log_info "Exists, skipping: BP${i} -> $(basename "$out_file")"
    ((++skip_existing_count))
    continue
  fi

  attempts=0
  while (( attempts < RETRIES )); do
    ((++attempts))
    tmp_file="$OUT_DIR/.BP${padded}.tmp$$.$RANDOM"

    log_info "GET $url (attempt ${attempts}/$RETRIES)"

    http_status=""
    curl_rc=0
    # Capture HTTP status; keep curl rc for transport errors
    if ! http_status=$(curl -sS -L \
      -A "$UA" \
      --connect-timeout 15 --max-time 60 \
      -o "$tmp_file" \
      -w '%{http_code}' \
      "$url" 2>>"$LOG_FILE"); then
      curl_rc=$?
    fi

    # Normalize status to integer if present, else 0
    if [[ -z "$http_status" || ! "$http_status" =~ ^[0-9]{3}$ ]]; then
      http_status=0
    fi

    if (( curl_rc == 0 )) && (( http_status >= 200 && http_status < 300 )) && [[ -s "$tmp_file" ]]; then
      mv -f -- "$tmp_file" "$out_file"
      log_info "Saved BP${i} -> $(basename "$out_file") [HTTP $http_status]"
      ((++success_count))
      break
    else
      # Failure path
      [[ -f "$tmp_file" ]] && rm -f -- "$tmp_file" || true
      if (( curl_rc != 0 )); then
        log_warn "Curl transport error for $url (rc=$curl_rc); attempt ${attempts}/$RETRIES"
      else
        if (( http_status == 0 )); then
          log_warn "No HTTP status/empty response for $url; attempt ${attempts}/$RETRIES"
        else
          log_warn "HTTP $http_status for $url; attempt ${attempts}/$RETRIES"
        fi
      fi

      if (( attempts < RETRIES )); then
        # Exponential backoff with jitter
        backoff=$(( 2 ** attempts ))
        jitter=$(rand_sleep)
        log_info "Retrying after ${backoff}s + ${jitter}s"
        sleep "$backoff"
        sleep "$jitter"
      else
        log_error "Giving up on $url after $RETRIES attempts; skipping"
        ((++fail_count))
      fi
    fi
  done

  # Be gentle: randomized delay between successful downloads, too
  if (( attempts <= RETRIES )) && [[ -s "$out_file" ]]; then
    delay=$(rand_sleep)
    log_info "Sleeping ${delay}s before next request"
    sleep "$delay"
  fi
done

log_info "Done. Success: $success_count, Skipped(existing): $skip_existing_count, Failed: $fail_count"

exit 0
