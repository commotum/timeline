Here are the most solid, “modern uv” patterns for building and using Python CLIs, with the tradeoffs and the uv commands you’ll actually use.

## 1) Pick the right delivery model for your CLI

### A. **Packaged CLI command (recommended for real tools)**

You publish/install a package that exposes a command like `mytool …`.

**How:** define an entry point in `pyproject.toml` under `[project.scripts]` (this is the standard packaging mechanism). uv’s docs explicitly treat these as *entry points* for CLIs. ([Astral Docs][1])

Key gotcha: **entry points require a build system** in `pyproject.toml` (e.g., hatchling/setuptools). Without it, uv won’t wire up your scripts the way you expect. ([Astral Docs][1])

Typical skeleton:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mytool"
version = "0.1.0"
dependencies = ["typer>=0.12"]  # or click/argparse, etc.

[project.scripts]
mytool = "mytool.cli:main"
```

Then:

* **Dev loop:** `uv run mytool --help` (runs inside the project env, and keeps it up to date) ([Astral Docs][2])
* **Build/publish:** `uv build` / `uv publish` when you’re ready to ship it ([Astral Docs][3])

### B. **Script-first CLI (great for internal utilities)**

If what you really want is “a runnable script” more than a distributable package, uv has excellent script workflows:

* `uv run --script your_script.py`
* or make it executable with a **shebang** so it runs directly on your PATH. ([Astral Docs][4])

This is ideal for: ops scripts, one-off automation, internal tooling, “single file” utilities.

### C. **Global “installed command” for your local project**

When you want your in-progress CLI to behave like a global command on your machine (without publishing it), use uv’s **tools** feature (pipx-like). ([Astral Docs][5])

A common workflow is:

* install your local project as a tool (often editable) so `mytool` works anywhere
* keep hacking, and the command reflects your changes

(There are many ways to do this; see section 3.)

---

## 2) Use `uv run` as your default dev runner

In a uv-managed project, the environment lives at `.venv`, and uv’s recommended way to run anything that needs the project installed is `uv run …` because it ensures the env is present and up to date. ([Astral Docs][2])

Practical dev pattern:

* `uv sync` (create/update venv + deps)
* `uv run mytool …` (exercise your CLI)
* `uv run python -m mytool …` (module execution variant)

---

## 3) “Global CLI installs”: `uv tool install` vs `uvx`

uv has a first-class concept of running/hosting CLI tools:

### `uvx` = one-off, ephemeral tool run

`uvx` is an alias for “run a tool in a disposable cached environment.” Great for trying CLIs or running pinned versions without “installing” them permanently. ([Astral Docs][5])

Examples from the docs:

* run a specific tool version: `uvx ruff@latest check` ([Astral Docs][6])

### `uv tool install` = keep a tool installed

If you want something on your PATH long-term (pipx-style), use `uv tool install …`. uv’s docs position this as “install and manage tools.” ([Astral Docs][5])

This is also a great way to install **your own local CLI** so you can run it from anywhere.

---

## 4) Shipping a CLI package cleanly (build + publish)

When you’re ready to distribute:

* `uv build` produces wheels/sdists
* `uv publish` uploads to a package index (PyPI or another) ([Astral Docs][3])

Two practical tips that bite people:

* Make sure your `pyproject.toml` is present in CI jobs that publish (uv’s publish guide calls this out). ([Astral Docs][3])
* Keep your CLI entry point in `[project.scripts]` and keep the callable stable (`pkg.module:function`), so users don’t break when you refactor internals.

---

## 5) Monorepos / multiple CLIs

If your repo will contain multiple packages (e.g., shared lib + several CLIs), uv has a **workspace** concept for managing multiple members together. ([Astral Docs][7])

Pattern:

* one workspace
* multiple packages, each with its own `[project.scripts]` exposing its CLI(s)
* shared deps handled at the workspace level as appropriate

---

## 6) CLI framework choice (uv-specific advice: keep it packaging-friendly)

uv doesn’t force a CLI framework; it just cares that your package installs cleanly and exposes entry points.

Common “works everywhere” setups:

* `argparse` (stdlib, simplest)
* `click` (very common)
* `typer` (nice ergonomics, type hints)

The uv-specific best practice is: **put CLI logic behind a `main()` function** and point `[project.scripts]` to it. That keeps packaging/entry points trivial and testable.

---

### A very solid “default” workflow (if you want one)

1. Add `[build-system]` + `[project.scripts]` (packaged CLI entry point) ([Astral Docs][1])
2. Use `uv run mytool …` during development ([Astral Docs][2])
3. For “global on my machine” installs, use `uv tool install` for persistence; use `uvx` for one-offs ([Astral Docs][5])
4. When ready, `uv build` + `uv publish` ([Astral Docs][3])

If you tell me which of these you’re aiming for:

* **internal script** vs **installable command** vs **publish to PyPI**,
  I’ll give you a concrete minimal `pyproject.toml` + folder layout + the exact uv commands for your preferred workflow.

[1]: https://docs.astral.sh/uv/concepts/projects/config/?utm_source=chatgpt.com "Configuring projects | uv - Astral Docs"
[2]: https://docs.astral.sh/uv/concepts/projects/run/?utm_source=chatgpt.com "Running commands in projects - uv - Astral Docs"
[3]: https://docs.astral.sh/uv/guides/package/?utm_source=chatgpt.com "Building and publishing a package | uv - Astral Docs"
[4]: https://docs.astral.sh/uv/guides/scripts/?utm_source=chatgpt.com "Running scripts | uv - Astral Docs"
[5]: https://docs.astral.sh/uv/?utm_source=chatgpt.com "uv - Astral Docs"
[6]: https://docs.astral.sh/uv/guides/tools/?utm_source=chatgpt.com "Using tools | uv - Astral Docs"
[7]: https://docs.astral.sh/uv/concepts/projects/workspaces/?utm_source=chatgpt.com "Using workspaces | uv - Astral Docs"
