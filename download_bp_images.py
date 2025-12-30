#!/usr/bin/env python3
"""
Bongard Problem image downloader
- Downloads URLs listed in bp_image_urls.csv
- Saves to oebp_images/... (mirrors URL paths)
- Sequential requests, randomized delay, retries with backoff, and logging

Usage:
    pip install requests tqdm colorama
    python scripts/download_bp_images.py

Notes:
- Configure via env vars if desired:
    CSV_PATH, URL_COLUMN, OUT_DIR, SLEEP_MIN, SLEEP_MAX, RETRIES, UA, LOG_FILE,
    TIMEOUT_CONNECT_S, TIMEOUT_READ_S, MAX_FILES, OVERWRITE
"""
from __future__ import annotations

import csv
import os
import random
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from colorama import Fore, Style
from tqdm import tqdm

# Defaults (can be overridden via env vars before running)
CSV_PATH = Path(os.getenv("CSV_PATH", "bp_image_urls.csv"))
URL_COLUMN = os.getenv("URL_COLUMN", "image_url")
OUT_DIR = Path(os.getenv("OUT_DIR", "oebp_images"))
SLEEP_MIN = float(os.getenv("SLEEP_MIN", "2"))
SLEEP_MAX = float(os.getenv("SLEEP_MAX", "6"))
RETRIES = int(os.getenv("RETRIES", "3"))
TIMEOUT_CONNECT_S = float(os.getenv("TIMEOUT_CONNECT_S", "15"))
TIMEOUT_READ_S = float(os.getenv("TIMEOUT_READ_S", "60"))
MAX_FILES = int(os.getenv("MAX_FILES", "0"))
OVERWRITE = os.getenv("OVERWRITE", "").lower() in {"1", "true", "yes", "y"}
UA = os.getenv(
    "UA",
    "arc-analysis-bongard-downloader/1.0 "
    "(script: download_bp_images.py; purpose: research archival; url: https://oebp.org)",
)
LOG_FILE = Path(os.getenv("LOG_FILE", str(OUT_DIR / "download_bp_images.log")))

HEADERS = {
    "User-Agent": UA,
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}


def usage() -> None:
    script = Path(sys.argv[0]).name
    print(
        f"""Usage: {script}

Downloads image URLs listed in:
  {CSV_PATH}

Saves files to:
  {OUT_DIR}

Notes:
- Configure via env vars if desired:
    CSV_PATH, URL_COLUMN, OUT_DIR, SLEEP_MIN, SLEEP_MAX, RETRIES, UA, LOG_FILE,
    TIMEOUT_CONNECT_S, TIMEOUT_READ_S, MAX_FILES, OVERWRITE
"""
    )


if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    usage()
    sys.exit(0)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(level: str, message: str, color: str | None = None) -> None:
    line = f"[{timestamp()}] [{level}] {message}"
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if color:
        line = f"{color}{line}{Style.RESET_ALL}"
    tqdm.write(line)


def log_info(message: str, color: str | None = None) -> None:
    log("INFO", message, color=color)


def log_warn(message: str) -> None:
    log("WARN", message, color=Fore.YELLOW)


def log_error(message: str) -> None:
    log("ERROR", message, color=Fore.RED)


def rand_sleep(min_s: float, max_s: float) -> float:
    min_ms = int(min_s * 1000)
    max_ms = int(max_s * 1000)
    if max_ms < min_ms:
        min_ms, max_ms = max_ms, min_ms
    if max_ms == min_ms:
        return min_ms / 1000.0
    delay_ms = random.randint(min_ms, max_ms)
    return delay_ms / 1000.0


def cleanup_tmp(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.rglob("*.part"):
        try:
            path.unlink()
        except OSError:
            pass


def load_urls(csv_path: Path, url_column: str) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or url_column not in reader.fieldnames:
            raise ValueError(
                f"CSV missing required column '{url_column}' (found: {reader.fieldnames})"
            )
        urls = [row[url_column].strip() for row in reader if row.get(url_column)]
    return list(dict.fromkeys(urls))


def output_path_for_url(output_dir: Path, url: str) -> Path:
    parsed = urlparse(url)
    path = parsed.path.lstrip("/")
    return output_dir / path


def download_with_retries(
    session: requests.Session,
    url: str,
    dest: Path,
) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, RETRIES + 1):
        tmp_path = dest.with_suffix(dest.suffix + ".part")
        log_info(f"GET {url} (attempt {attempt}/{RETRIES})")
        try:
            with session.get(
                url,
                timeout=(TIMEOUT_CONNECT_S, TIMEOUT_READ_S),
                stream=True,
                allow_redirects=True,
            ) as resp:
                status = resp.status_code
                if 200 <= status < 300:
                    with tmp_path.open("wb") as handle:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                handle.write(chunk)
                    tmp_path.replace(dest)
                    log_info(f"Saved {url} -> {dest} [HTTP {status}]")
                    return True

                log_warn(f"HTTP {status} for {url}; attempt {attempt}/{RETRIES}")
        except requests.RequestException as exc:
            log_warn(f"Request error for {url}: {exc}; attempt {attempt}/{RETRIES}")
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        if attempt < RETRIES:
            backoff = 2 ** attempt
            jitter = rand_sleep(SLEEP_MIN, SLEEP_MAX)
            log_info(f"Retrying after {backoff}s + {jitter:.3f}s")
            time.sleep(backoff)
            time.sleep(jitter)

    return False


def main() -> int:
    if not CSV_PATH.exists():
        log_error(f"Missing CSV: {CSV_PATH}")
        return 1

    try:
        urls = load_urls(CSV_PATH, URL_COLUMN)
    except Exception as exc:
        log_error(f"Failed to read CSV: {exc}")
        return 1

    if MAX_FILES > 0:
        urls = urls[:MAX_FILES]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    cleanup_tmp(OUT_DIR)

    session = requests.Session()
    session.headers.update(HEADERS)

    success_count = 0
    skip_existing_count = 0
    fail_count = 0

    log_info(f"Starting image download: {CSV_PATH} -> {OUT_DIR}")
    log_info(f"User-Agent: {UA}")

    for url in tqdm(urls, unit="img", desc="Downloading"):
        dest = output_path_for_url(OUT_DIR, url)
        if dest.exists() and dest.stat().st_size > 0 and not OVERWRITE:
            log_info(f"Exists, skipping: {url} -> {dest}", color=Fore.CYAN)
            skip_existing_count += 1
            continue

        ok = download_with_retries(session, url, dest)
        if ok:
            success_count += 1
            delay = rand_sleep(SLEEP_MIN, SLEEP_MAX)
            log_info(f"Sleeping {delay:.3f}s before next request")
            time.sleep(delay)
        else:
            fail_count += 1

    log_info(
        "Done. "
        f"Success: {success_count}, "
        f"Skipped(existing): {skip_existing_count}, "
        f"Failed: {fail_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
