#!/usr/bin/env python3
import argparse
import csv
import datetime
import os
import re
import subprocess
import sys
import time
from pathlib import Path


STEP_ORDER = [
    "precheck",
    "download_pdfs",
    "ingest_local",
    "extract",
    "initial_classification",
    "transfer",
]


def format_cmd(cmd):
    return " ".join(str(part) for part in cmd)


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log_event(log_path, step, event, message=""):
    line = f"{timestamp()} step={step} event={event}"
    if message:
        line = f"{line} {message}"
    print(f"[pipeline] {step}: {event}{' ' + message if message else ''}")
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def run_step(name, cmd, cwd, env, continue_on_error, log_path):
    log_event(log_path, name, "start")
    log_event(log_path, name, "cmd", f"cmd={format_cmd(cmd)}")
    start = time.time()
    try:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
    except FileNotFoundError as exc:
        log_event(log_path, name, "failed", f"missing_executable={exc}")
        if continue_on_error:
            return False
        raise
    except subprocess.CalledProcessError as exc:
        log_event(log_path, name, "failed", f"exit={exc.returncode}")
        if continue_on_error:
            return False
        raise
    elapsed = time.time() - start
    log_event(log_path, name, "done", f"elapsed={elapsed:.1f}s")
    return True


def ensure_path(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the NEW pipeline: download, extract, classify, transfer."
    )
    parser.add_argument(
        "--skip-precheck",
        action="store_true",
        help="Skip precheck.py.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download_pdfs.sh.",
    )
    parser.add_argument(
        "--skip-ingest-local",
        action="store_true",
        help="Skip ingesting NEW/Local PDFs and LOCAL.csv.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extract.py.",
    )
    parser.add_argument(
        "--skip-classify",
        action="store_true",
        help="Skip initial classification.",
    )
    parser.add_argument(
        "--skip-transfer",
        action="store_true",
        help="Skip transfer.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to classification and transfer steps.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later steps if a step fails.",
    )
    return parser.parse_args()


def detect_resume_step(log_path, steps):
    if not log_path or not log_path.exists():
        return None

    step_re = re.compile(r"\bstep=([a-z_]+)\b")
    event_re = re.compile(r"\bevent=([a-z_]+)\b")
    last_event = {}

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            step_match = step_re.search(line)
            event_match = event_re.search(line)
            if not step_match or not event_match:
                continue
            step = step_match.group(1)
            event = event_match.group(1)
            if step in steps:
                last_event[step] = event

    for step in steps:
        event = last_event.get(step)
        if event in ("start", "failed"):
            return step

    return None


def sanitize_title(value):
    sanitized = value
    for ch in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]:
        sanitized = sanitized.replace(ch, "-")
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def load_csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def write_csv_atomic(path, fieldnames, rows):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temp_path.replace(path)


def ingest_local(local_dir, local_csv, downloads_dir, new_csv, log_path):
    if not local_dir.exists() or not local_csv.exists():
        log_event(log_path, "ingest_local", "skipped", "reason=missing_local")
        return True

    try:
        local_fields, local_rows = load_csv_rows(local_csv)
    except Exception as exc:
        log_event(log_path, "ingest_local", "failed", f"local_csv_error={exc}")
        return False

    if not local_rows:
        log_event(log_path, "ingest_local", "skipped", "reason=empty_local_csv")
        return True

    try:
        new_fields, new_rows = load_csv_rows(new_csv)
    except Exception as exc:
        log_event(log_path, "ingest_local", "failed", f"new_csv_error={exc}")
        return False

    if not new_fields:
        log_event(log_path, "ingest_local", "failed", "new_csv_missing_header")
        return False

    required = {"year", "title", "url"}
    if not required.issubset(set(local_fields)):
        log_event(log_path, "ingest_local", "failed", "local_csv_missing_columns")
        return False

    downloads_dir.mkdir(parents=True, exist_ok=True)

    existing_titles = {
        (row.get("title") or "").strip().lower() for row in new_rows if row.get("title")
    }
    existing_urls = {
        (row.get("url") or "").strip().lower() for row in new_rows if row.get("url")
    }

    pdf_map = {}
    pdf_dupes = set()
    for item in local_dir.iterdir():
        if not item.is_file() or item.suffix.lower() != ".pdf":
            continue
        key = item.stem.lower()
        if key in pdf_map:
            pdf_dupes.add(key)
        else:
            pdf_map[key] = item

    if pdf_dupes:
        log_event(log_path, "ingest_local", "failed", "duplicate_local_pdfs")
        return False

    appended = 0
    moved = 0
    skipped_moves = 0
    missing_files = 0

    for row in local_rows:
        title = (row.get("title") or "").strip()
        url = (row.get("url") or "").strip()
        year = (row.get("year") or "").strip()
        if not title:
            continue
        title_key = title.lower()
        url_key = url.lower() if url else ""

        if title_key in existing_titles or (url_key and url_key in existing_urls):
            existing_titles.add(title_key)
            if url_key:
                existing_urls.add(url_key)
        else:
            new_rows.append({"year": year, "title": title, "url": url})
            appended += 1
            existing_titles.add(title_key)
            if url_key:
                existing_urls.add(url_key)

        expected_name = sanitize_title(title)
        pdf = pdf_map.get(expected_name.lower())
        if not pdf:
            missing_files += 1
            continue
        dest = downloads_dir / f"{expected_name}.pdf"
        if dest.exists():
            skipped_moves += 1
            continue
        pdf.rename(dest)
        moved += 1

    try:
        write_csv_atomic(new_csv, new_fields, new_rows)
    except Exception as exc:
        log_event(log_path, "ingest_local", "failed", f"new_csv_write_error={exc}")
        return False

    try:
        write_csv_atomic(local_csv, local_fields, [])
    except Exception as exc:
        log_event(log_path, "ingest_local", "failed", f"local_csv_clear_error={exc}")
        return False

    log_event(
        log_path,
        "ingest_local",
        "done",
        f"appended={appended} moved={moved} skipped_moves={skipped_moves} missing_files={missing_files}",
    )
    return True


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    new_root = repo_root / "NEW"
    scripts_dir = new_root / "Scripts"

    download_script = scripts_dir / "download_pdfs.sh"
    extract_script = scripts_dir / "extract.py"
    transfer_script = scripts_dir / "transfer.py"
    runner_script = (
        repo_root / "RUNNERS" / "initial-classification" / "script-initial-classification.py"
    )
    precheck_script = scripts_dir / "precheck.py"

    csv_path = new_root / "NEW.csv"
    downloads_dir = new_root / "Downloads"
    markdown_dir = new_root / "Markdown"
    local_dir = new_root / "Local"
    local_csv = local_dir / "LOCAL.csv"
    bib_csv_path = repo_root / "BIBLIOTHEQUE" / "BIBLIOTHEQUE.csv"
    pipeline_log = scripts_dir / "pipeline.log"

    try:
        ensure_path(download_script, "download script")
        ensure_path(precheck_script, "precheck script")
        ensure_path(extract_script, "extract script")
        ensure_path(transfer_script, "transfer script")
        ensure_path(runner_script, "classification runner")
        ensure_path(csv_path, "NEW.csv")
        ensure_path(bib_csv_path, "BIBLIOTHEQUE.csv")
        ensure_path(downloads_dir, "Downloads dir")
        ensure_path(markdown_dir, "Markdown dir")
    except FileNotFoundError as exc:
        print(f"[pipeline] setup failed: {exc}")
        return 1

    env = os.environ.copy()
    cwd = str(repo_root)

    resume_from = detect_resume_step(pipeline_log, STEP_ORDER)
    if resume_from:
        log_event(
            pipeline_log,
            "pipeline",
            "resume",
            f"resume_from={resume_from}",
        )
    else:
        log_event(pipeline_log, "pipeline", "start")

    all_ok = True

    steps = list(STEP_ORDER)
    if resume_from:
        resume_index = steps.index(resume_from)
        steps = steps[resume_index:]

    if "precheck" in steps and not args.skip_precheck:
        ok = run_step(
            "precheck",
            [sys.executable, str(precheck_script)],
            cwd,
            env,
            args.continue_on_error,
            pipeline_log,
        )
        all_ok = all_ok and ok
    elif "precheck" in steps and args.skip_precheck:
        log_event(pipeline_log, "precheck", "skipped", "reason=flag")

    if "download_pdfs" in steps and not args.skip_download:
        ok = run_step(
            "download_pdfs",
            ["bash", str(download_script), str(csv_path), str(downloads_dir)],
            cwd,
            env,
            args.continue_on_error,
            pipeline_log,
        )
        all_ok = all_ok and ok
    elif "download_pdfs" in steps and args.skip_download:
        log_event(pipeline_log, "download_pdfs", "skipped", "reason=flag")

    if "ingest_local" in steps and not args.skip_ingest_local:
        ok = ingest_local(
            local_dir,
            local_csv,
            downloads_dir,
            csv_path,
            pipeline_log,
        )
        all_ok = all_ok and ok
    elif "ingest_local" in steps and args.skip_ingest_local:
        log_event(pipeline_log, "ingest_local", "skipped", "reason=flag")

    if "extract" in steps and not args.skip_extract:
        ok = run_step(
            "extract",
            [sys.executable, str(extract_script)],
            cwd,
            env,
            args.continue_on_error,
            pipeline_log,
        )
        all_ok = all_ok and ok
    elif "extract" in steps and args.skip_extract:
        log_event(pipeline_log, "extract", "skipped", "reason=flag")

    if "initial_classification" in steps and not args.skip_classify:
        cmd = [
            "uv",
            "run",
            "python",
            str(runner_script),
            "--folder",
            str(markdown_dir),
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        ok = run_step(
            "initial_classification",
            cmd,
            cwd,
            env,
            args.continue_on_error,
            pipeline_log,
        )
        all_ok = all_ok and ok
    elif "initial_classification" in steps and args.skip_classify:
        log_event(pipeline_log, "initial_classification", "skipped", "reason=flag")

    if "transfer" in steps and not args.skip_transfer:
        cmd = [sys.executable, str(transfer_script)]
        if args.dry_run:
            cmd.append("--dry-run")
        ok = run_step(
            "transfer",
            cmd,
            cwd,
            env,
            args.continue_on_error,
            pipeline_log,
        )
        all_ok = all_ok and ok
    elif "transfer" in steps and args.skip_transfer:
        log_event(pipeline_log, "transfer", "skipped", "reason=flag")

    if not all_ok:
        log_event(pipeline_log, "pipeline", "completed", "status=errors")
        return 1

    log_event(pipeline_log, "pipeline", "completed", "status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
