TARGET_FOLDERS = [
    "BIBLIOTHEQUE/01_POS-ENCDR",
    "BIBLIOTHEQUE/02_XFORM-DIM",
]
PROMPT_PATH = "prompt-extending-dimensions.md"
OUTPUT_FILENAME = "EXTENDING_DIMENSIONS.md"
CODEX_CLI_CMD = "codex"
DRY_RUN = False
OVERWRITE_MD = False
SORT_MODE = "alpha"
PARALLEL_WORKERS = 4
LOG_PATH = "codex_extending_dimensions.log"
CODEX_EXEC_ARGS = ["exec", "--full-auto"]  # Adjust Codex CLI mode/permissions here.
CODEX_EXEC_TIMEOUT = 3600
CODEX_CWD = None
CODEX_ADD_DIR = True
CODEX_SKIP_GIT_CHECK = False
MAX_LOG_CHARS = 2000
MIN_MD_BYTES = 1
CLEANUP_TEMP_ON_START = True

import argparse
import datetime
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError

from tqdm import tqdm

LOG_LOCK = threading.Lock()


class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.count = 0
        self._bar = tqdm(total=total, unit="file", dynamic_ncols=True, ascii=True)

    def update(self, name, status):
        self.count += 1
        label = f"{status} {name}"
        self._bar.update(1)
        self._bar.set_postfix_str(label, refresh=True)

    def write(self, message):
        self._bar.write(message)

    def close(self):
        self._bar.close()


def progress_write(progress, message):
    if progress is None:
        print(message)
    else:
        progress.write(message)


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log_event(log_path, message):
    line = f"{now_iso()} {message}\n"
    try:
        with LOG_LOCK:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(line)
    except Exception:
        sys.stderr.write(line)


def load_state(state_path, log_path):
    if not os.path.exists(state_path):
        return {"completed": {}}
    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("state is not a dict")
        if "completed" not in data or not isinstance(data["completed"], dict):
            data["completed"] = {}
        return data
    except Exception as exc:
        log_event(log_path, f"state_load_failed path={state_path} error={exc}")
        return {"completed": {}}


def save_state(state_path, state, log_path):
    try:
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
    except Exception as exc:
        log_event(log_path, f"state_save_failed path={state_path} error={exc}")


def find_git_root(start_path):
    current = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def resolve_path(path, base_dirs):
    if os.path.isabs(path):
        return os.path.abspath(path)
    for base in base_dirs:
        if not base:
            continue
        candidate = os.path.abspath(os.path.join(base, path))
        if os.path.exists(candidate):
            return candidate
    base = next((b for b in base_dirs if b), os.getcwd())
    return os.path.abspath(os.path.join(base, path))


def sanitize_log_text(text, max_chars):
    if not text:
        return ""
    scrubbed = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(scrubbed) <= max_chars:
        return scrubbed
    return scrubbed[:max_chars] + "...(truncated)"


def quote_path(path):
    if any(ch.isspace() for ch in path):
        return '"' + path.replace('"', '\\"') + '"'
    return path


def build_prompt(
    template, source_md_abs_path, md_abs_path, file_stem, output_folder
):
    prompt = template
    replacements = {
        "[SOURCE_MD_ABS_PATH]": quote_path(source_md_abs_path),
        "[MD_ABS_PATH]": quote_path(md_abs_path),
        "[FILE_STEM]": file_stem,
        "[OUTPUT_FOLDER]": quote_path(output_folder),
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def safe_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


def list_paper_entries(target_folders, output_name, log_path):
    entries = []
    for class_folder in target_folders:
        if not os.path.isdir(class_folder):
            log_event(log_path, f"missing_target_folder path={class_folder}")
            continue
        for name in os.listdir(class_folder):
            paper_folder = os.path.join(class_folder, name)
            if not os.path.isdir(paper_folder):
                continue
            file_stem = name
            source_md = os.path.join(paper_folder, f"{file_stem}.md")
            if not os.path.isfile(source_md):
                log_event(
                    log_path,
                    f"missing_source_md folder={paper_folder} path={source_md}",
                )
                continue
            output_md = os.path.join(paper_folder, output_name)
            entries.append(
                {
                    "name": file_stem,
                    "folder": paper_folder,
                    "source_md": os.path.abspath(source_md),
                    "output_md": os.path.abspath(output_md),
                }
            )
    return entries


def sort_queue(entries, sort_mode, log_path):
    mode = (sort_mode or "").strip().lower()
    if mode == "alpha":
        return sorted(entries, key=lambda e: e["name"].lower())
    if mode in ("mtime", "mtime-desc", "newest"):
        return sorted(
            entries,
            key=lambda e: safe_mtime(e["source_md"]),
            reverse=True,
        )
    if mode in ("mtime-asc", "oldest"):
        return sorted(entries, key=lambda e: safe_mtime(e["source_md"]))
    log_event(log_path, f"unknown_sort_mode mode={sort_mode} fallback=alpha")
    return sorted(entries, key=lambda e: e["name"].lower())


def ensure_md_file(md_abs_path, dry_run, log_path):
    if dry_run:
        log_event(log_path, f"dry_run_md_prepare path={md_abs_path}")
        return True
    try:
        with open(md_abs_path, "w", encoding="utf-8") as handle:
            handle.write("")
        return True
    except Exception as exc:
        log_event(log_path, f"md_prepare_failed path={md_abs_path} error={exc}")
        return False


def make_temp_md_path(output_path, output_name):
    folder = os.path.dirname(output_path)
    prefix = f".{output_name}.tmp."
    while True:
        temp_name = f"{prefix}{uuid.uuid4().hex}"
        temp_path = os.path.join(folder, temp_name)
        if not os.path.exists(temp_path):
            return temp_path


def remove_temp_md(path, log_path):
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
            log_event(log_path, f"temp_removed path={path}")
    except Exception as exc:
        log_event(log_path, f"temp_remove_failed path={path} error={exc}")


def cleanup_temp_paths(temp_paths, log_path):
    if not temp_paths:
        return
    for path in list(temp_paths):
        remove_temp_md(path, log_path)
        temp_paths.discard(path)


def cleanup_stale_temp_files(folders, output_name, log_path):
    prefix = f".{output_name}.tmp."
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        removed = 0
        try:
            for name in os.listdir(folder):
                if not name.startswith(prefix):
                    continue
                path = os.path.join(folder, name)
                if not os.path.isfile(path):
                    continue
                try:
                    os.remove(path)
                    removed += 1
                    log_event(log_path, f"temp_removed path={path}")
                except Exception as exc:
                    log_event(log_path, f"temp_remove_failed path={path} error={exc}")
        except Exception as exc:
            log_event(log_path, f"temp_cleanup_failed folder={folder} error={exc}")
            continue
        if removed:
            log_event(log_path, f"temp_cleanup_removed folder={folder} count={removed}")


def build_codex_exec_command(codex_cmd, codex_cwd, add_dir, skip_git_check):
    args = shlex.split(codex_cmd)
    if not args:
        return []
    if "exec" not in args:
        args.extend(CODEX_EXEC_ARGS)
    if codex_cwd and "-C" not in args and "--cd" not in args:
        args.extend(["-C", codex_cwd])
    if add_dir and "--add-dir" not in args:
        args.extend(["--add-dir", add_dir])
    if skip_git_check and "--skip-git-repo-check" not in args:
        args.append("--skip-git-repo-check")
    return args


def run_codex_exec(
    codex_cmd,
    prompt,
    codex_cwd,
    add_dir,
    skip_git_check,
    log_path,
    stop_event=None,
):
    args = build_codex_exec_command(codex_cmd, codex_cwd, add_dir, skip_git_check)
    if not args:
        return False, "empty_codex_cmd"
    if stop_event and stop_event.is_set():
        log_event(log_path, "codex_skip_interrupted")
        return False, "interrupted"
    base_cmd = args[0]
    if shutil.which(base_cmd) is None:
        log_event(log_path, f"codex_missing cmd={base_cmd}")
        return False, "codex_missing"
    cmd_display = " ".join(shlex.quote(arg) for arg in args)
    log_event(log_path, f"codex_launch cmd={cmd_display}")
    start_time = time.monotonic()
    try:
        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        log_event(log_path, f"codex_exception error={exc}")
        return False, str(exc)

    stdout = ""
    stderr = ""
    input_payload = prompt
    try:
        while True:
            if stop_event and stop_event.is_set():
                proc.terminate()
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                elapsed = time.monotonic() - start_time
                log_event(log_path, f"codex_interrupted seconds={elapsed:.1f}")
                return False, "interrupted"
            try:
                stdout, stderr = proc.communicate(input=input_payload, timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                input_payload = None
                if time.monotonic() - start_time > CODEX_EXEC_TIMEOUT:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                    elapsed = time.monotonic() - start_time
                    log_event(log_path, f"codex_timeout seconds={elapsed:.1f}")
                    return False, "timeout"
                continue
    except Exception as exc:
        try:
            proc.kill()
            proc.communicate()
        except Exception:
            pass
        log_event(log_path, f"codex_exception error={exc}")
        return False, str(exc)

    elapsed = time.monotonic() - start_time
    if proc.returncode != 0:
        stderr_trimmed = sanitize_log_text(stderr, MAX_LOG_CHARS)
        stdout_trimmed = sanitize_log_text(stdout, MAX_LOG_CHARS)
        log_event(
            log_path,
            f"codex_failure rc={proc.returncode} seconds={elapsed:.1f} stderr={stderr_trimmed} stdout={stdout_trimmed}",
        )
        return False, f"rc={proc.returncode}"
    log_event(log_path, f"codex_success seconds={elapsed:.1f}")
    return True, None


def md_has_content(md_abs_path):
    try:
        return os.path.getsize(md_abs_path) >= MIN_MD_BYTES
    except OSError:
        return False


def remove_existing_output(path, log_path):
    try:
        if os.path.exists(path):
            os.remove(path)
            log_event(log_path, f"output_removed path={path}")
    except Exception as exc:
        log_event(log_path, f"output_remove_failed path={path} error={exc}")


def process_task(
    source_md_abs_path,
    prompt,
    codex_cmd,
    codex_cwd,
    add_dir,
    skip_git_check,
    log_path,
    stop_event,
):
    success, error = run_codex_exec(
        codex_cmd,
        prompt,
        codex_cwd,
        add_dir,
        skip_git_check,
        log_path,
        stop_event,
    )
    return success, error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Codex CLI for a queue of OCR markdown files."
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="One or more class folders to scan for paper subfolders.",
    )
    parser.add_argument(
        "--prompt-path",
        default=PROMPT_PATH,
        help="Prompt template markdown file.",
    )
    parser.add_argument(
        "--output-name",
        default=OUTPUT_FILENAME,
        help="Output markdown filename per paper folder.",
    )
    parser.add_argument(
        "--codex-cmd",
        default=CODEX_CLI_CMD,
        help="Command to launch Codex CLI.",
    )
    parser.add_argument(
        "--sort-mode",
        default=None,
        help="Sort mode: alpha, mtime, mtime-asc, mtime-desc, newest, oldest.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel Codex jobs.",
    )
    parser.add_argument(
        "--skip-git-check",
        action="store_true",
        help="Skip Codex git repo check.",
    )
    dry_group = parser.add_mutually_exclusive_group()
    dry_group.add_argument("--dry-run", dest="dry_run", action="store_true")
    dry_group.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=None)
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument("--overwrite", dest="overwrite", action="store_true")
    overwrite_group.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=None)
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = find_git_root(script_dir) or find_git_root(os.getcwd())
    base_dirs = [os.getcwd(), script_dir, repo_root]

    target_folders = args.folders if args.folders else TARGET_FOLDERS
    resolved_folders = [resolve_path(path, base_dirs) for path in target_folders]

    prompt_path = resolve_path(args.prompt_path, base_dirs)
    if not os.path.exists(prompt_path):
        print(f"Prompt not found: {prompt_path}")
        return 1
    with open(prompt_path, "r", encoding="utf-8") as handle:
        prompt_template = handle.read()

    output_name = args.output_name or OUTPUT_FILENAME
    if not output_name.lower().endswith(".md"):
        output_name += ".md"

    dry_run = DRY_RUN if args.dry_run is None else args.dry_run
    overwrite = OVERWRITE_MD if args.overwrite is None else args.overwrite
    sort_mode = SORT_MODE if args.sort_mode is None else args.sort_mode
    workers = PARALLEL_WORKERS if args.workers is None else args.workers
    if workers < 1:
        workers = 1
    codex_cmd = args.codex_cmd

    log_path = LOG_PATH
    if not os.path.isabs(log_path):
        log_path = os.path.abspath(os.path.join(script_dir, log_path))
    state_path = os.path.join(script_dir, ".codex_extending_dimensions_state.json")

    log_event(
        log_path,
        "run_start "
        f"folders={resolved_folders} dry_run={dry_run} overwrite={overwrite} "
        f"sort_mode={sort_mode} workers={workers} output_name={output_name}",
    )

    codex_cwd = os.path.abspath(CODEX_CWD) if CODEX_CWD else None
    if codex_cwd is None:
        codex_cwd = repo_root
    if codex_cwd:
        log_event(log_path, f"codex_cwd path={codex_cwd}")
    else:
        log_event(log_path, "codex_cwd_missing")
    add_dir = repo_root if CODEX_ADD_DIR else None
    skip_git_check = CODEX_SKIP_GIT_CHECK or args.skip_git_check

    state = load_state(state_path, log_path)
    completed = state.get("completed", {})

    entries = list_paper_entries(resolved_folders, output_name, log_path)
    queue = sort_queue(entries, sort_mode, log_path)

    total = len(queue)
    if total == 0:
        print("No OCR markdown files found.")
        log_event(log_path, "no_entries_found")
        log_event(log_path, "run_end")
        return 0

    progress = ProgressTracker(total)
    stop_event = threading.Event()
    temp_paths = set()
    futures = {}
    executor = None
    interrupted = False

    try:
        if CLEANUP_TEMP_ON_START and not dry_run:
            cleanup_stale_temp_files(
                [entry["folder"] for entry in queue], output_name, log_path
            )

        if not dry_run:
            executor = ThreadPoolExecutor(max_workers=workers)

        for entry in queue:
            if stop_event.is_set():
                interrupted = True
                break

            file_stem = entry["name"]
            entry_folder = entry["folder"]
            source_md_abs_path = entry["source_md"]
            output_md_abs_path = entry["output_md"]

            log_event(
                log_path,
                f"file_start source_md={source_md_abs_path} output_md={output_md_abs_path}",
            )

            if not overwrite and source_md_abs_path in completed:
                log_event(log_path, f"skipped_state source_md={source_md_abs_path}")
                progress.update(file_stem, "skipped")
                continue

            if not overwrite and os.path.exists(output_md_abs_path):
                log_event(
                    log_path,
                    f"skipped_existing_output source_md={source_md_abs_path}",
                )
                if not dry_run:
                    completed[source_md_abs_path] = now_iso()
                    state["completed"] = completed
                    save_state(state_path, state, log_path)
                progress.update(file_stem, "skipped")
                continue

            if overwrite and os.path.exists(output_md_abs_path) and not dry_run:
                remove_existing_output(output_md_abs_path, log_path)

            md_work_path = make_temp_md_path(output_md_abs_path, output_name)
            if not dry_run:
                if not ensure_md_file(md_work_path, dry_run, log_path):
                    log_event(
                        log_path,
                        f"skipped_md_prepare_failed source_md={source_md_abs_path}",
                    )
                    progress.update(file_stem, "failed")
                    continue
                temp_paths.add(md_work_path)
                log_event(
                    log_path,
                    f"md_work_created source_md={source_md_abs_path} md_work={md_work_path}",
                )

            prompt = build_prompt(
                prompt_template,
                source_md_abs_path,
                md_work_path,
                file_stem,
                entry_folder,
            )
            log_event(
                log_path,
                f"prompt_generated source_md={source_md_abs_path} md_work={md_work_path}",
            )

            if dry_run:
                progress_write(progress, "DRY RUN prompt:")
                progress_write(progress, prompt)
                log_event(log_path, f"dry_run_skip source_md={source_md_abs_path}")
                progress.update(file_stem, "dry-run")
                continue

            future = executor.submit(
                process_task,
                source_md_abs_path,
                prompt,
                codex_cmd,
                codex_cwd,
                add_dir,
                skip_git_check,
                log_path,
                stop_event,
            )
            futures[future] = (
                source_md_abs_path,
                file_stem,
                md_work_path,
                output_md_abs_path,
            )

        if executor is not None:
            for future in as_completed(futures):
                source_md_abs_path, file_stem, md_work_path, output_md_abs_path = (
                    futures[future]
                )
                try:
                    success, error = future.result()
                except CancelledError:
                    log_event(log_path, f"codex_cancelled source_md={source_md_abs_path}")
                    remove_temp_md(md_work_path, log_path)
                    temp_paths.discard(md_work_path)
                    progress.update(file_stem, "cancelled")
                    continue
                except Exception as exc:
                    log_event(
                        log_path,
                        f"codex_failure source_md={source_md_abs_path} error={exc}",
                    )
                    remove_temp_md(md_work_path, log_path)
                    temp_paths.discard(md_work_path)
                    progress.update(file_stem, "failed")
                    continue
                if success:
                    if not md_has_content(md_work_path):
                        log_event(log_path, f"md_empty source_md={source_md_abs_path}")
                        progress_write(
                            progress,
                            f"  warning: output file is empty for {file_stem} (see log)",
                        )
                        remove_temp_md(md_work_path, log_path)
                        temp_paths.discard(md_work_path)
                        progress.update(file_stem, "empty")
                        continue
                    try:
                        os.replace(md_work_path, output_md_abs_path)
                    except Exception as exc:
                        log_event(
                            log_path,
                            f"md_replace_failed source_md={source_md_abs_path} error={exc}",
                        )
                        remove_temp_md(md_work_path, log_path)
                        temp_paths.discard(md_work_path)
                        progress.update(file_stem, "failed")
                        continue
                    temp_paths.discard(md_work_path)
                    log_event(
                        log_path,
                        f"md_written source_md={source_md_abs_path} md_final={output_md_abs_path}",
                    )
                    completed[source_md_abs_path] = now_iso()
                    state["completed"] = completed
                    save_state(state_path, state, log_path)
                    progress.update(file_stem, "done")
                else:
                    log_event(
                        log_path,
                        f"codex_failure source_md={source_md_abs_path} error={error}",
                    )
                    remove_temp_md(md_work_path, log_path)
                    temp_paths.discard(md_work_path)
                    if error == "codex_missing":
                        progress_write(
                            progress,
                            "Codex CLI not found. Paste this prompt into Codex manually:",
                        )
                        prompt_final = build_prompt(
                            prompt_template,
                            source_md_abs_path,
                            output_md_abs_path,
                            file_stem,
                            os.path.dirname(output_md_abs_path),
                        )
                        progress_write(progress, prompt_final)
                        log_event(
                            log_path,
                            f"manual_prompt_printed source_md={source_md_abs_path}",
                        )
                    status = "failed"
                    if error == "interrupted":
                        status = "interrupted"
                    progress.update(file_stem, status)
            executor.shutdown(wait=True)
            executor = None
    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
        log_event(log_path, "run_interrupted")
        progress_write(progress, "Interrupted. Cancelling pending tasks...")
    except Exception as exc:
        stop_event.set()
        log_event(log_path, f"run_exception error={exc}")
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        cleanup_temp_paths(temp_paths, log_path)
        if progress is not None:
            progress.close()
        log_event(log_path, "run_end")

    if interrupted:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
