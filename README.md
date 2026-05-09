# Repo Doctor Agent

A small but complete MVP for a **codebase understanding + issue fixing Agent**.

It turns a local bug/issue description into:

1. a repair plan,
2. relevant file selection,
3. a unified diff patch,
4. patch review,
5. optional patch application,
6. optional test execution and one repair round,
7. a PR summary.

The default mode is safe: it prints a patch and does **not** change files unless you pass `--apply`.

## Setup

```bash
cd repo_doctor_agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-5.5"  # or another model you have access to
```

## Dry run on a repo

```bash
python -m repo_doctor_agent.agent \
  --repo /path/to/your/repo \
  --issue "Login fails when password contains #. Expected login to succeed, actual response is 400." \
  --write-patch fix.patch
```

## Apply patch and run tests

```bash
python -m repo_doctor_agent.agent \
  --repo /path/to/your/repo \
  --issue-file issue.md \
  --apply \
  --test-command "python -m pytest -q"
```

## Recommended workflow

1. Commit or stash your current work first.
2. Run a dry run and inspect the patch.
3. Re-run with `--apply` only when the patch looks safe.
4. Review the diff manually before opening a PR.

## Notes

- The tool uses `git apply --check` before applying a patch.
- It never pushes code or opens a PR by itself.
- It excludes common large folders such as `.git`, `node_modules`, `dist`, `build`, and `target`.
- You can tune context limits with `--max-file-chars` and `--max-context-chars`.
