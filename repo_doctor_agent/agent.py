#!/usr/bin/env python3
"""
Repo Doctor Agent: a small but complete Issue -> Patch -> Test -> PR-summary agent.

What it does:
1. Reads an issue/bug description.
2. Indexes a local repository.
3. Asks a Planner Agent which files matter.
4. Asks a Code Agent to produce a unified diff patch.
5. Checks/applies the patch with git apply.
6. Runs tests, optionally asks the agent for one repair round.
7. Generates a PR summary.

Default mode is safe: it prints the patch but does not modify files unless --apply is passed.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
    "target",
    "coverage",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    ".turbo",
    ".venv",
    "venv",
    "env",
    ".tox",
}

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".tgz",
    ".7z",
    ".mp3",
    ".mp4",
    ".mov",
    ".avi",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".bin",
}

CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".cs",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sql",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".txt",
    ".html",
    ".css",
    ".scss",
    ".sass",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".dockerfile",
}

SPECIAL_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "package.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    "build.gradle",
    "settings.gradle",
    "README.md",
}


@dataclass
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float

    @property
    def combined_output(self) -> str:
        chunks = []
        if self.stdout.strip():
            chunks.append("STDOUT:\n" + self.stdout.strip())
        if self.stderr.strip():
            chunks.append("STDERR:\n" + self.stderr.strip())
        return "\n\n".join(chunks).strip()


class AgentError(RuntimeError):
    pass


class OpenAIResponsesClient:
    """Tiny raw-HTTP client for the OpenAI Responses API.

    This avoids depending on a specific openai Python SDK version. Set OPENAI_API_KEY
    in your environment before running.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, base_url: str = OPENAI_BASE_URL):
        if not api_key:
            raise AgentError("OPENAI_API_KEY is missing. Run: export OPENAI_API_KEY='your_key'")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        max_output_tokens: int = 4096,
        temperature: Optional[float] = 0.2,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": user}]},
            ],
            "max_output_tokens": max_output_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        response = self._post("/responses", payload)
        return self._extract_output_text(response).strip()

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_output_tokens: int = 4096,
    ) -> Dict[str, Any]:
        text = self.complete_text(
            system=system + "\n\nReturn valid JSON only. Do not include markdown fences.",
            user=user,
            max_output_tokens=max_output_tokens,
            temperature=0.1,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
            if match:
                return json.loads(match.group(1))
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise AgentError(f"Model did not return valid JSON. Raw output:\n{text[:2000]}")

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url + path
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)
        except requests.RequestException as exc:
            raise AgentError(f"OpenAI request failed: {exc}") from exc

        # Some reasoning models may reject temperature. Retry once without it.
        if r.status_code >= 400 and "temperature" in payload and "temperature" in r.text.lower():
            payload = dict(payload)
            payload.pop("temperature", None)
            r = requests.post(url, headers=headers, json=payload, timeout=120)

        if r.status_code >= 400:
            raise AgentError(f"OpenAI API error {r.status_code}: {r.text[:2000]}")
        return r.json()

    @staticmethod
    def _extract_output_text(data: Dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str):
            return data["output_text"]

        parts: List[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                value_type = value.get("type")
                text = value.get("text")
                if value_type in {"output_text", "text"} and isinstance(text, str):
                    parts.append(text)
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(data.get("output", []))
        if parts:
            return "\n".join(parts)
        raise AgentError(f"Could not extract text from API response: {json.dumps(data)[:2000]}")


class RepoDoctorAgent:
    def __init__(
        self,
        repo: Path,
        client: OpenAIResponsesClient,
        max_file_chars: int = 18_000,
        max_context_chars: int = 95_000,
    ):
        self.repo = repo.resolve()
        self.client = client
        self.max_file_chars = max_file_chars
        self.max_context_chars = max_context_chars
        if not self.repo.exists() or not self.repo.is_dir():
            raise AgentError(f"Repository path does not exist or is not a directory: {self.repo}")

    def run(
        self,
        *,
        issue: str,
        apply_patch: bool,
        yes: bool,
        test_command: Optional[str],
        skip_tests: bool,
        max_fix_rounds: int,
        write_patch_path: Optional[Path],
    ) -> None:
        print_step("1/6 Indexing repository")
        tree = self.collect_tree(max_entries=1200)
        candidate_files = self.list_candidate_files(max_files=1500)
        print(f"Found {len(candidate_files)} candidate text/code files.\n")

        print_step("2/6 Planning")
        plan = self.plan(issue=issue, tree=tree)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        files = self._sanitize_files(plan.get("files_to_read") or [], candidate_files)
        if not files:
            files = self.heuristic_relevant_files(issue, candidate_files)[:10]
        print("\nFiles selected:")
        for f in files:
            print(f"- {f}")

        print_step("3/6 Generating patch")
        context = self.build_context(files)
        patch = self.generate_patch(issue=issue, plan=plan, context=context)
        patch = clean_patch(patch)
        if not patch:
            raise AgentError("The model returned an empty patch.")

        if write_patch_path:
            write_patch_path.write_text(patch, encoding="utf-8")
            print(f"Patch written to {write_patch_path}")

        print("\n--- GENERATED PATCH ---\n")
        print(patch)

        print_step("4/6 Reviewing patch")
        review = self.review_patch(issue=issue, plan=plan, patch=patch)
        print(json.dumps(review, ensure_ascii=False, indent=2))

        check = self.git_apply_check(patch)
        if check.returncode != 0:
            print("\nPatch check failed:\n" + check.combined_output)
            print("\nAsking agent for a corrected patch...")
            patch = clean_patch(
                self.generate_patch(
                    issue=issue,
                    plan=plan,
                    context=context,
                    previous_patch=patch,
                    failure_output=check.combined_output,
                )
            )
            check = self.git_apply_check(patch)
            if check.returncode != 0:
                raise AgentError("Corrected patch still failed git apply --check:\n" + check.combined_output)
            print("Corrected patch passes git apply --check.")

        if not apply_patch:
            print_step("5/6 Dry run complete")
            print("Patch was not applied. Re-run with --apply to modify files.")
            summary = self.generate_pr_summary(issue=issue, plan=plan, patch=patch, test_result=None)
            print_step("6/6 PR summary")
            print(summary)
            return

        if not yes:
            confirm = input("\nApply this patch to the repo? Type 'yes' to continue: ").strip().lower()
            if confirm != "yes":
                print("Cancelled. No files changed.")
                return

        apply_result = self.git_apply(patch)
        if apply_result.returncode != 0:
            raise AgentError("git apply failed:\n" + apply_result.combined_output)
        print("Patch applied successfully.")

        test_result: Optional[CommandResult] = None
        if not skip_tests:
            print_step("5/6 Running tests")
            detected_command = test_command or self.detect_test_command()
            if not detected_command:
                print("No test command detected. Pass --test-command to run your test suite.")
            else:
                test_result = run_command(detected_command, cwd=self.repo, timeout_seconds=300)
                print_command_result(test_result)

                repair_round = 0
                while test_result.returncode != 0 and repair_round < max_fix_rounds:
                    repair_round += 1
                    print(f"\nTests failed. Repair round {repair_round}/{max_fix_rounds}...")
                    repair_context = self.build_context(files)
                    repair_patch = clean_patch(
                        self.generate_patch(
                            issue=issue,
                            plan=plan,
                            context=repair_context,
                            previous_patch=patch,
                            failure_output=test_result.combined_output,
                        )
                    )
                    if not repair_patch:
                        print("Repair agent returned no patch. Stopping repair loop.")
                        break
                    check2 = self.git_apply_check(repair_patch)
                    if check2.returncode != 0:
                        print("Repair patch did not apply cleanly:\n" + check2.combined_output)
                        break
                    apply2 = self.git_apply(repair_patch)
                    if apply2.returncode != 0:
                        print("Repair patch apply failed:\n" + apply2.combined_output)
                        break
                    patch += "\n\n" + repair_patch
                    test_result = run_command(detected_command, cwd=self.repo, timeout_seconds=300)
                    print_command_result(test_result)
        else:
            print_step("5/6 Tests skipped")

        print_step("6/6 PR summary")
        summary = self.generate_pr_summary(issue=issue, plan=plan, patch=patch, test_result=test_result)
        print(summary)

    def plan(self, *, issue: str, tree: str) -> Dict[str, Any]:
        system = """
You are Planner Agent for a repository maintenance tool. Your job is to analyze an issue and decide which files should be inspected.
Return JSON with these keys:
- summary: short issue summary
- root_cause_hypothesis: likely cause
- files_to_read: array of repository-relative paths from the tree only, maximum 12 files
- steps: array of concrete repair steps
- test_strategy: how to validate the fix
- risk: low | medium | high
Do not invent file paths that are not in the tree.
""".strip()
        user = f"""
ISSUE:
{issue}

REPOSITORY TREE:
{tree}
""".strip()
        return self.client.complete_json(system=system, user=user, max_output_tokens=2400)

    def generate_patch(
        self,
        *,
        issue: str,
        plan: Dict[str, Any],
        context: str,
        previous_patch: Optional[str] = None,
        failure_output: Optional[str] = None,
    ) -> str:
        system = """
You are Code Agent. Produce a minimal, correct unified diff patch for the repository.
Rules:
1. Return ONLY a unified diff, no markdown, no explanation.
2. Use repository-relative paths exactly.
3. Prefer small, targeted changes.
4. Add or update tests when feasible.
5. Do not remove unrelated code.
6. If there is not enough context to safely modify the repo, return an empty string.
""".strip()
        extra = ""
        if previous_patch:
            extra += f"\n\nPREVIOUS PATCH:\n{previous_patch[:30000]}"
        if failure_output:
            extra += f"\n\nFAILURE OUTPUT TO FIX:\n{failure_output[:30000]}"
        user = f"""
ISSUE:
{issue}

PLAN:
{json.dumps(plan, ensure_ascii=False, indent=2)}

REPOSITORY CONTEXT:
{context}
{extra}
""".strip()
        return self.client.complete_text(system=system, user=user, max_output_tokens=9000, temperature=0.1)

    def review_patch(self, *, issue: str, plan: Dict[str, Any], patch: str) -> Dict[str, Any]:
        system = """
You are Reviewer Agent. Review whether the patch is relevant, safe, and testable.
Return JSON with keys:
- accept: boolean
- severity: low | medium | high
- issues: array of concrete concerns
- test_notes: suggested tests
- pr_notes: concise PR note
""".strip()
        user = f"""
ISSUE:
{issue}

PLAN:
{json.dumps(plan, ensure_ascii=False, indent=2)}

PATCH:
{patch[:50000]}
""".strip()
        return self.client.complete_json(system=system, user=user, max_output_tokens=2200)

    def generate_pr_summary(
        self,
        *,
        issue: str,
        plan: Dict[str, Any],
        patch: str,
        test_result: Optional[CommandResult],
    ) -> str:
        system = """
You are PR Summary Agent. Write a concise pull request description in Markdown.
Include:
- What changed
- Why
- How it was tested
- Risk/rollback notes
""".strip()
        test_text = "Tests were not run."
        if test_result:
            status = "passed" if test_result.returncode == 0 else "failed"
            test_text = f"Command: {test_result.command}\nStatus: {status}\nOutput:\n{test_result.combined_output[:12000]}"
        user = f"""
ISSUE:
{issue}

PLAN:
{json.dumps(plan, ensure_ascii=False, indent=2)}

PATCH:
{patch[:50000]}

TEST RESULT:
{test_text}
""".strip()
        return self.client.complete_text(system=system, user=user, max_output_tokens=2600, temperature=0.2)

    def collect_tree(self, max_entries: int = 1200) -> str:
        lines: List[str] = []
        count = 0
        for path in sorted(self.repo.rglob("*")):
            if count >= max_entries:
                lines.append("... tree truncated ...")
                break
            rel = path.relative_to(self.repo)
            if self._is_ignored(rel, is_dir=path.is_dir()):
                continue
            depth = len(rel.parts) - 1
            prefix = "  " * depth
            marker = "/" if path.is_dir() else ""
            lines.append(f"{prefix}{rel.name}{marker}")
            count += 1
        return "\n".join(lines)

    def list_candidate_files(self, max_files: int = 1500) -> List[str]:
        files: List[str] = []
        for path in sorted(self.repo.rglob("*")):
            if len(files) >= max_files:
                break
            if not path.is_file():
                continue
            rel = path.relative_to(self.repo)
            if self._is_ignored(rel, is_dir=False):
                continue
            if is_probably_text_code_file(path):
                files.append(str(rel).replace(os.sep, "/"))
        return files

    def heuristic_relevant_files(self, issue: str, candidate_files: Sequence[str]) -> List[str]:
        tokens = [t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", issue)]
        scored: List[Tuple[int, str]] = []
        for f in candidate_files:
            name = f.lower()
            score = 0
            for t in tokens:
                if t in name:
                    score += 5
            if any(part in name for part in ["test", "spec"]):
                score += 1
            if Path(f).name in SPECIAL_FILENAMES:
                score += 1
            if score > 0:
                scored.append((score, f))
        scored.sort(reverse=True)
        return [f for _, f in scored]

    def build_context(self, files: Sequence[str]) -> str:
        chunks: List[str] = []
        total = 0
        for rel in files:
            path = self.repo / rel
            if not path.exists() or not path.is_file():
                continue
            text = read_text_safely(path, max_chars=self.max_file_chars)
            chunk = f"\n<file path=\"{rel}\">\n{text}\n</file>\n"
            if total + len(chunk) > self.max_context_chars:
                chunks.append("\n<!-- context truncated because max_context_chars was reached -->\n")
                break
            chunks.append(chunk)
            total += len(chunk)
        return "".join(chunks)

    def detect_test_command(self) -> Optional[str]:
        if (self.repo / "pytest.ini").exists() or (self.repo / "pyproject.toml").exists() or (self.repo / "tests").exists():
            if any((self.repo / p).exists() for p in ["pytest.ini", "tests"]):
                return "python -m pytest -q"
        package_json = self.repo / "package.json"
        if package_json.exists():
            try:
                data = json.loads(package_json.read_text(encoding="utf-8"))
                scripts = data.get("scripts") or {}
                if "test" in scripts:
                    return "npm test"
            except Exception:
                return "npm test"
        if (self.repo / "go.mod").exists():
            return "go test ./..."
        if (self.repo / "Cargo.toml").exists():
            return "cargo test"
        if (self.repo / "pom.xml").exists():
            return "mvn test"
        if (self.repo / "build.gradle").exists() or (self.repo / "settings.gradle").exists():
            return "./gradlew test"
        return None

    def git_apply_check(self, patch: str) -> CommandResult:
        return run_command_with_stdin(
            ["git", "apply", "--check", "--whitespace=fix", "-"],
            stdin=patch,
            cwd=self.repo,
            timeout_seconds=60,
        )

    def git_apply(self, patch: str) -> CommandResult:
        return run_command_with_stdin(
            ["git", "apply", "--whitespace=fix", "-"],
            stdin=patch,
            cwd=self.repo,
            timeout_seconds=60,
        )

    def _sanitize_files(self, requested: Sequence[str], candidate_files: Sequence[str]) -> List[str]:
        valid = set(candidate_files)
        clean: List[str] = []
        for item in requested:
            rel = str(item).strip().strip("./")
            rel = rel.replace("\\", "/")
            if rel in valid and rel not in clean:
                clean.append(rel)
        return clean[:12]

    def _is_ignored(self, rel: Path, is_dir: bool) -> bool:
        parts = set(rel.parts)
        if parts & IGNORED_DIRS:
            return True
        name = rel.name
        if name.startswith(".") and name not in {".env.example", ".gitignore", ".dockerignore"}:
            return True
        for pattern in ["*.min.js", "*.lock", "*.map"]:
            if fnmatch.fnmatch(name, pattern):
                return True
        if not is_dir and rel.suffix.lower() in BINARY_EXTENSIONS:
            return True
        return False


def is_probably_text_code_file(path: Path) -> bool:
    if path.name in SPECIAL_FILENAMES:
        return True
    if path.suffix.lower() in CODE_EXTENSIONS:
        return True
    return False


def read_text_safely(path: Path, max_chars: int) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return f"<!-- could not read file: {exc} -->"
    if b"\x00" in raw[:4096]:
        return "<!-- binary file skipped -->"
    for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return "<!-- could not decode file -->"
    if len(text) > max_chars:
        return text[:max_chars] + f"\n<!-- file truncated at {max_chars} characters -->"
    return text


def clean_patch(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:diff|patch)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    idx = text.find("diff --git ")
    if idx >= 0:
        text = text[idx:].strip()
    # Remove accidental explanation after the patch if the model adds a common marker.
    for marker in ["\nExplanation:", "\nNotes:", "\nPR Summary:"]:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def run_command(command: str, cwd: Path, timeout_seconds: int) -> CommandResult:
    start = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        return CommandResult(command, proc.returncode, proc.stdout, proc.stderr, time.time() - start)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(command, 124, exc.stdout or "", exc.stderr or f"Timed out after {timeout_seconds}s", time.time() - start)


def run_command_with_stdin(command: Sequence[str], stdin: str, cwd: Path, timeout_seconds: int) -> CommandResult:
    start = time.time()
    pretty = " ".join(shlex.quote(c) for c in command)
    try:
        proc = subprocess.run(
            list(command),
            input=stdin,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        return CommandResult(pretty, proc.returncode, proc.stdout, proc.stderr, time.time() - start)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(pretty, 124, exc.stdout or "", exc.stderr or f"Timed out after {timeout_seconds}s", time.time() - start)


def print_step(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_command_result(result: CommandResult) -> None:
    print(f"$ {result.command}")
    print(f"exit={result.returncode} elapsed={result.elapsed_seconds:.1f}s")
    if result.combined_output:
        print(result.combined_output[:20000])


def read_issue(args: argparse.Namespace) -> str:
    if args.issue_file:
        return Path(args.issue_file).read_text(encoding="utf-8")
    if args.issue:
        return args.issue
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise AgentError("Provide --issue, --issue-file, or pipe issue text through stdin.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repo Doctor Agent: turn a bug issue into a patch, tests, and PR summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo", default=".", help="Path to the local repository")
    parser.add_argument("--issue", help="Issue/bug description")
    parser.add_argument("--issue-file", help="Path to a file containing the issue description")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--apply", action="store_true", help="Apply the generated patch to the repo")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation before applying")
    parser.add_argument("--skip-tests", action="store_true", help="Do not run tests after applying")
    parser.add_argument("--test-command", help="Custom test command, e.g. 'pytest -q' or 'npm test'")
    parser.add_argument("--max-fix-rounds", type=int, default=1, help="Repair rounds after test failure")
    parser.add_argument("--write-patch", type=Path, help="Optional path to save the generated patch")
    parser.add_argument("--max-file-chars", type=int, default=18_000, help="Max chars read from each file")
    parser.add_argument("--max-context-chars", type=int, default=95_000, help="Max total chars sent as repo context")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        issue = read_issue(args).strip()
        if not issue:
            raise AgentError("Issue text is empty.")
        client = OpenAIResponsesClient(api_key=os.environ.get("OPENAI_API_KEY", ""), model=args.model)
        agent = RepoDoctorAgent(
            repo=Path(args.repo),
            client=client,
            max_file_chars=args.max_file_chars,
            max_context_chars=args.max_context_chars,
        )
        agent.run(
            issue=issue,
            apply_patch=args.apply,
            yes=args.yes,
            test_command=args.test_command,
            skip_tests=args.skip_tests,
            max_fix_rounds=max(0, args.max_fix_rounds),
            write_patch_path=args.write_patch,
        )
        return 0
    except AgentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
