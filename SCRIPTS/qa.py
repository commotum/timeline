TARGET_FOLDER = "/home/jake/Developer/timeline/CHUNKED"
PROMPT_PATH = "/home/jake/Developer/timeline/PROMPTS/QUALITY-ASSURANCE.md"
GLOSSARY_PATH = "/home/jake/Developer/timeline/PROMPTS/DATA-QUALITY-PROBLEMS.md"
CODEX_CLI_CMD = "codex"
CODEX_EXEC_ARGS = ["exec", "--full-auto"]
CODEX_EXEC_TIMEOUT = 3600
CODEX_CWD = None
CODEX_ADD_DIR = True
CODEX_SKIP_GIT_CHECK = False
LOG_PATH = "codex_qa.log"
SORT_MODE = "alpha"
DRY_RUN = False

import argparse
import datetime
import os
import shlex
import shutil
import subprocess
import sys
import time


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log_event(log_path, message):
    line = f"{now_iso()} {message}\n"
    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        sys.stderr.write(line)


def find_git_root(start_path):
    current = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def quote_path(path):
    if any(ch.isspace() for ch in path):
        return '"' + path.replace('"', '\\"') + '"'
    return path


def list_csv_files(target_folder):
    entries = []
    for name in os.listdir(target_folder):
        if not name.lower().endswith(".csv"):
            continue
        full_path = os.path.join(target_folder, name)
        if os.path.isfile(full_path):
            entries.append(full_path)
    return entries


def sort_queue(entries, sort_mode, log_path):
    mode = (sort_mode or "").strip().lower()
    if mode == "alpha":
        return sorted(entries, key=lambda p: os.path.basename(p).lower())
    if mode in ("mtime", "mtime-desc", "newest"):
        return sorted(entries, key=lambda p: os.path.getmtime(p), reverse=True)
    if mode in ("mtime-asc", "oldest"):
        return sorted(entries, key=lambda p: os.path.getmtime(p))
    log_event(log_path, f"unknown_sort_mode mode={sort_mode} fallback=alpha")
    return sorted(entries, key=lambda p: os.path.basename(p).lower())


def build_prompt(template_path, source_path, glossary_path):
    with open(template_path, "r", encoding="utf-8") as handle:
        template = handle.read()
    prompt = template.replace("[SOURCE_CSV_PATH]", quote_path(source_path))
    prompt = prompt.replace("[GLOSSARY_PATH]", quote_path(glossary_path))
    return prompt


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
):
    args = build_codex_exec_command(codex_cmd, codex_cwd, add_dir, skip_git_check)
    if not args:
        return False, "empty_codex_cmd"
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
    try:
        stdout, stderr = proc.communicate(input=prompt, timeout=CODEX_EXEC_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        elapsed = time.monotonic() - start_time
        log_event(log_path, f"codex_timeout seconds={elapsed:.1f}")
        return False, "timeout"
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
        log_event(
            log_path,
            f"codex_failure rc={proc.returncode} seconds={elapsed:.1f}",
        )
        return False, f"rc={proc.returncode}"
    log_event(log_path, f"codex_success seconds={elapsed:.1f}")
    return True, None


def ensure_glossary_file(glossary_path, dry_run, log_path):
    if dry_run:
        log_event(log_path, f"dry_run_glossary_prepare path={glossary_path}")
        return True
    try:
        if not os.path.exists(glossary_path):
            with open(glossary_path, "w", encoding="utf-8") as handle:
                handle.write("")
        return True
    except Exception as exc:
        log_event(
            log_path,
            f"glossary_prepare_failed path={glossary_path} error={exc}",
        )
        return False


def resolve_source_path(source_arg, target_folder):
    if not source_arg:
        return None
    if os.path.isabs(source_arg):
        return source_arg
    candidate = os.path.join(target_folder, source_arg)
    if os.path.exists(candidate):
        return candidate
    return os.path.abspath(source_arg)


def main():
    parser = argparse.ArgumentParser(
        description="Run Codex QA prompt over a single chunked CSV file.",
    )
    parser.add_argument("--target-folder", default=TARGET_FOLDER)
    parser.add_argument("--prompt-path", default=PROMPT_PATH)
    parser.add_argument("--glossary-path", default=GLOSSARY_PATH)
    parser.add_argument("--source", default=None)
    parser.add_argument("--sort-mode", default=SORT_MODE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--codex-cmd", default=CODEX_CLI_CMD)
    parser.add_argument("--skip-git-check", action="store_true")
    args = parser.parse_args()

    target_folder = os.path.abspath(args.target_folder)
    prompt_path = os.path.abspath(args.prompt_path)
    glossary_path = os.path.abspath(args.glossary_path)
    log_path = LOG_PATH
    if not os.path.isabs(log_path):
        log_path = os.path.abspath(os.path.join(target_folder, log_path))

    if not os.path.isdir(target_folder):
        print(f"Target folder not found: {target_folder}")
        return 1
    if not os.path.exists(prompt_path):
        print(f"Prompt not found: {prompt_path}")
        return 1

    dry_run = DRY_RUN or args.dry_run
    log_event(
        log_path,
        f"run_start folder={target_folder} dry_run={dry_run} sort_mode={args.sort_mode}",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    codex_cwd = os.path.abspath(CODEX_CWD) if CODEX_CWD else None
    if codex_cwd is None:
        codex_cwd = find_git_root(script_dir) or find_git_root(target_folder)
    if codex_cwd:
        log_event(log_path, f"codex_cwd path={codex_cwd}")
    else:
        log_event(log_path, "codex_cwd_missing")

    add_dir = target_folder if CODEX_ADD_DIR else None
    skip_git_check = CODEX_SKIP_GIT_CHECK or args.skip_git_check

    if not ensure_glossary_file(glossary_path, dry_run, log_path):
        print(f"Failed to prepare glossary file: {glossary_path}")
        return 1

    source_path = resolve_source_path(args.source, target_folder)
    if source_path is None:
        sources = sort_queue(list_csv_files(target_folder), args.sort_mode, log_path)
        if not sources:
            print("No CSV files found.")
            log_event(log_path, "no_csv_found")
            log_event(log_path, "run_end")
            return 0
        if len(sources) > 1:
            print("Multiple CSV files found; pass --source to select one.")
            log_event(log_path, "multiple_csv_found")
            log_event(log_path, "run_end")
            return 1
        source_path = sources[0]

    source_path = os.path.abspath(source_path)
    if not os.path.exists(source_path):
        print(f"Source CSV not found: {source_path}")
        log_event(log_path, f"source_missing path={source_path}")
        log_event(log_path, "run_end")
        return 1

    prompt = build_prompt(prompt_path, source_path, glossary_path)
    log_event(log_path, f"file_start source={source_path}")
    print(os.path.basename(source_path))
    if dry_run:
        log_event(log_path, f"dry_run_skip source={source_path}")
        log_event(log_path, "run_end")
        return 0

    success, error = run_codex_exec(
        args.codex_cmd,
        prompt,
        codex_cwd,
        add_dir,
        skip_git_check,
        log_path,
    )
    if success:
        log_event(log_path, f"file_success source={source_path}")
    else:
        log_event(log_path, f"file_failure source={source_path} error={error}")

    log_event(log_path, "run_end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
