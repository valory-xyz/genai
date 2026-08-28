# genai

Open Autonomy connection and protocol packages (Gemini, OpenAI, x402).
`packages/packages.json` holds two protocols and three connections — there are no
agent or service packages here.

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
| `.github/workflows/common_checks.yaml` | six `pip install 'tomte[tox,cli] @ git+…'` lines |
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

- **`check-generate-all-protocols`** shows up in `tomte tox -l`, is run by no
  Valory repo, and must not be added. Regenerating strips the copyright headers
  that `tomte check-copyright --author valory` then demands, and the generator
  stamps the current year, so it would go red every 1 January. It also needs
  `protoc` and `protolint`, which no CI installs.
- **`packages/valory/connections/x402`** is adapted from coinbase/x402 and is
  deliberately held outside the lint, type-hint and copyright surface — see
  `service_specific_packages_exclude`, `--exclude-part x402`, and
  `[mypy-packages.valory.connections.x402.*]`. Leave its lint alone.
- **No service packages**, so anything keyed off one is a no-op or a trap here.
  tomte's `analyse-service` env was one; the release workflow's `--author`
  derivation was another.
