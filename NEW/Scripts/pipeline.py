#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import subprocess
import sys
import time
from pathlib import Path


STEP_ORDER = [
    "download_pdfs",
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
        "--skip-download",
        action="store_true",
        help="Skip download_pdfs.sh.",
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


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    new_root = repo_root / "NEW"
    scripts_dir = new_root / "Scripts"

    download_script = scripts_dir / "download_pdfs.sh"
    extract_script = scripts_dir / "extract.py"
    transfer_script = scripts_dir / "transfer.py"
    runner_script = (
        repo_root / "RUNNERS" / "initial-classification" / "script-initial-classification.py"
    )

    csv_path = new_root / "NEW.csv"
    downloads_dir = new_root / "Downloads"
    markdown_dir = new_root / "Markdown"
    pipeline_log = scripts_dir / "pipeline.log"

    try:
        ensure_path(download_script, "download script")
        ensure_path(extract_script, "extract script")
        ensure_path(transfer_script, "transfer script")
        ensure_path(runner_script, "classification runner")
        ensure_path(csv_path, "NEW.csv")
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
