# genai

Open Autonomy connection and protocol packages (Gemini, OpenAI, x402).
`packages/packages.json` holds protocols and connections only — there are no
agent or service packages here.

**Keep this file current.** If you hit a trap here that cost you time — a command
that does not behave as documented, a check that fails for reasons unrelated to
your change, a local failure CI never sees — add it below before you finish. The
next agent has no memory of your run. Keep entries to durable facts about this
repo; leave version-specific mechanics to the commit and PR that changed them,
where `git blame` can still reach them.

## Environment

This is a uv repo, not Poetry.

```bash
uv sync --all-groups
```

`uv.lock` is authoritative, and CI runs `uv lock --check` on Linux, macOS and
Windows. Any edit to `[project].dependencies` or `[dependency-groups]` has to be
followed by `uv lock`, or the `lock_check` job fails on all three.

## Python version

genai pins **3.14** for its lock, copyright/dependency and lint jobs. The sibling
Valory repos (trader, kv-store, optimus) pin 3.10 for those same jobs, so do not
carry a 3.10 assumption over from them. The test matrix is 3.10–3.14; the
gitleaks `scan` job is the one place genai still uses 3.10.

## tomte

tomte is the task runner — every check target shells into it (the `clean*` and
`push-packages` targets do not). Its version is pinned in three places that must
move together, plus the lockfile:

| Where | What |
|---|---|
| `pyproject.toml` | the `dev` dependency, and `[tool.tomte] tomte_dep_pin` |
| `.github/workflows/common_checks.yaml` | every `pip install 'tomte[tox,cli] @ git+…'` line |
| `uv.lock` | regenerate with `uv lock` |

The CI `pip install` is the driver that renders the tox config; `tomte_dep_pin`
only governs what each rendered env then installs. Bumping `pyproject.toml`
alone leaves the run that actually gates PRs on the old renderer.

`tox.ini` is not a tox config. It is a `[tomte-extensions]` overlay that
`tomte tox` merges into a canonical template, rendering `.tomte-tox.ini` at run
time. Invoke envs as `tomte tox -e <env>`; plain `tox -e <env>` will not work.

## Verification

```bash
make format           # rewrites files
make code-checks
make common-checks-1
make common-checks-2
```

Prefer those over `make all-checks`, which also runs `generators` and `security`
and rewrites files as a side effect. `make security` needs a `gitleaks` binary on
PATH; CI installs one, local shells usually do not.

After touching anything under `packages/`, run `make generators` — it calls
`autonomy packages lock`, which rewrites the IPFS hashes in
`packages/packages.json`. Skip it and `check-hash` fails.

## Repo-specific traps

- **Do not add `check-generate-all-protocols`, and do not run a bare `tomte tox`
  with no `-e`** — the env sits in tomte's default envlist. It rewrites the
  protocol files in place *before* checking them, so running it leaves your
  working tree dirty, and its output cannot satisfy `check-copyright`. It was
  investigated and rejected; the reasoning is in the git history.
- **`packages/valory/connections/x402`** is adapted from coinbase/x402 and is
  deliberately held outside the lint, type-hint and copyright surface — see
  `service_specific_packages_exclude`, `--exclude-part x402`, and
  `[mypy-packages.valory.connections.x402.*]`. Leave its lint alone.
- **No service packages**, so anything keyed off one is a no-op or a trap here.
- **`release.yaml`'s `set -eu` and its `--author` derivation are deliberate and
  interdependent.** The derivation has to yield a valid author handle or
  `autonomy init` fails, and `set -e` is what stops such a failure being masked
  by the command after it. Do not "simplify" either, and do not swap `-eu` for
  `-euo pipefail` — the action runs `sh`, and there are no pipelines in that
  block. The reasoning is in the PR that introduced them.
