#!/usr/bin/env python3
"""test-speed: the blocking check that keeps pull-request tests as short as they can be (test-speed-standard-20261122).

Miguel 2026-11-22 22:53 CT: "Test should always last the least amount of time possible ... make it the standard systemwide and enforce it hard".

    test_speed_check.py [--root .] [--no-wall] [--runs-json FILE | --repo OWNER/NAME --branch BRANCH --pr-number N [--current-run ID]]

The same file, byte for byte, sits in every repository that has CI (hermes-home owns it; tools/ci_speed_alarm.py compares the copies).
It reads `.github/workflows/*.y*ml` and fails (exit 1) when a workflow that runs on pull requests and runs tests:
  chain       chains test stages as separate runner jobs (`needs:` from one test job to another): every hop waits for a runner again;
              a recorded reason, `TEST_SPEED_CHAIN_REASON: "<at least 20 characters>"` in the dependent job's (or the workflow's) `env`, allows one;
  concurrency has no `concurrency:` whose group is per pull request or branch (per pull request for `pull_request_target`) and that cancels the
              older run (`cancel-in-progress` true for pull requests; a group with an unconditional run id never cancels);
  full-suite  runs every test on every pull-request push with no selector: a selecting tool or command in a `run` or `uses` of the job (a step name
              or a comment does not count), a test command that names its target (a file, a node id, `-k`), a path filter on every pull-request
              trigger, or a `needs.changes` style gate; or it records `TEST_SPEED_FULL_SUITE_REASON: "... measured N min ..."` with 0 < N <= budget.
What counts as a test stage: a test command (pytest, unittest, npm test, go test, cargo test, nix flake check, make test ...) in any `&&`/`;`/`|`
segment of a `run` that is not an install command, a local composite action or reusable workflow whose own steps run tests, or a job that marks
itself `TEST_SPEED_TEST_JOB: "true"` in its `env` (a wrapper script the check cannot see into).
Then the time budget. The check measures the latest finished, not cancelled, pull-request run of each workflow file of THIS pull request that still
exists in the tree: its execution time (first job start to last job end, so queue time is not charged here; the alarm owns queue time). Over the repo
budget (`.test-speed.json` `budget_min`, default 20; above 20 it needs a `budget_reason` with "measured") is `wall-time`; a workflow may carry its own
budget in `workflows.<name>` with a reason that says "measured". Exit 2: the tree or the config cannot be read (fail closed).
Limits, by design: a wrapper script with no marker is invisible; a GitHub outage while reading run history leaves the static rules deciding.
Prior art (RULES section 6): dorny/paths-filter and tj-actions/changed-files select by path and pytest-testmon by coverage, but none of them
fails a workflow that has no selector, chains stages or never cancels; this is a small stdlib-plus-PyYAML linter for that.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

try:
    import yaml
except ImportError:  # fail closed: a check that cannot read YAML must not pass
    yaml = None

DEFAULT_BUDGET = 20
PR_EVENTS = ("pull_request", "pull_request_target")
CONFIG_NAME = ".test-speed.json"
TEST_RE = re.compile(
    r"(?<![\w./-])(?:pytest|py\.test)\b|-m\s+(?:pytest|unittest)\b|\b(?:npm|pnpm|yarn)\s+(?:run\s+)?test\b|\bgo\s+test\b"
    r"|\bcargo\s+(?:test|nextest)\b|\bnix\s+flake\s+check\b|\bmake\s+test\b|\bctest\b|\bvitest\b|\bjest\b")
INSTALL_RE = re.compile(r"\b(?:pip3?|uv\s+pip|npm|pnpm|yarn|apt(?:-get)?|apk|brew|cargo)\s+(?:install|add)\b|\bpip\s+download\b")
SEGMENT_RE = re.compile(r"&&|\|\||[;|\n]")
SELECT_RE = re.compile(
    r"ci_select|select_tests|test_select|paths-filter|changed-files|testmon|--picked|git\s+diff\s+(?:\S+\s+)*--name-(?:only|status)"
    r"|needs\.(?:changes|detect|filter|paths)\b|steps\.(?:changes|select|filter)\b|needs\.[\w-]+\.outputs\.tests\b")
TARGET_RE = re.compile(r"\S+\.py\b|::|\s-k\s")
GROUP_RE = re.compile(r"github\.(?:ref|head_ref|event\.pull_request)|github\.event\.number")
TARGET_GROUP_RE = re.compile(r"pull_request\.number|head_ref|github\.event\.number")
RUN_ID_RE = re.compile(r"run_id|run_number|run_attempt")
MEASURED_RE = re.compile(r"measured\s+(\d+(?:\.\d+)?)\s*min", re.I)
FIX = {
    "chain": "merge the stages into one job (steps) so the run waits for a runner once, or record TEST_SPEED_CHAIN_REASON in the dependent job's env",
    "concurrency": "add `concurrency: {group: <name>-${{ github.ref }}, cancel-in-progress: ${{ github.event_name == 'pull_request' }}}` "
                   "(`github.event.pull_request.number` instead of `github.ref` for pull_request_target)",
    "full-suite": "run only what the change touches (tools/ci_select.py, git diff --name-only, a paths filter) or record TEST_SPEED_FULL_SUITE_REASON "
                  "\"... measured N min ...\" with N inside the budget",
}


class Finding(NamedTuple):
    rule: str
    where: str
    message: str


def epoch(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def code_lines(run) -> list[str]:
    """The command lines of a `run:` script without comment-only lines and trailing comments."""
    out = []
    for line in str(run).splitlines():
        line = re.sub(r"(^|\s)#.*$", "", line).strip()
        if line:
            out.append(line)
    return out


def has_test_command(run) -> bool:
    for line in code_lines(run):
        for segment in SEGMENT_RE.split(line):
            if TEST_RE.search(segment) and not INSTALL_RE.search(segment):
                return True
    return False


def steps_of(job: dict) -> list[dict]:
    steps = job.get("steps")
    return [s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []


def reason_of(job: dict, key: str, top_env: dict | None = None) -> str:
    for env in (job.get("env"), top_env):
        if isinstance(env, dict) and env.get(key) is not None:
            return str(env[key]).strip()
    return ""


def uses_of(node: dict) -> str:
    return node["uses"] if isinstance(node.get("uses"), str) else ""


def is_test_job(job: dict, local: dict | None = None, top_env: dict | None = None) -> bool:
    local = local or {}
    if reason_of(job, "TEST_SPEED_TEST_JOB", top_env).lower() == "true":  # a job that runs its tests through a wrapper script says so
        return True
    if local.get(uses_of(job), {}).get("test"):
        return True
    for step in steps_of(job):
        if has_test_command(step.get("run") or ""):
            return True
        if local.get(uses_of(step), {}).get("test"):
            return True
    return False


def selector_text(job: dict, local: dict | None = None) -> str:
    """What the job does, not what it says: run lines, `uses`, `if`, step env and with values; names and comments are left out."""
    local = local or {}
    env = job.get("env") if isinstance(job.get("env"), dict) else {}
    parts = [str(job.get("if", "")), uses_of(job), local.get(uses_of(job), {}).get("text", ""),
             json.dumps({k: v for k, v in env.items() if not str(k).startswith("TEST_SPEED_")}, default=str)]
    for step in steps_of(job):
        parts += [str(step.get("if", "")), uses_of(step), "\n".join(code_lines(step.get("run") or "")),
                  json.dumps(step.get("env", ""), default=str), json.dumps(step.get("with", ""), default=str), local.get(uses_of(step), {}).get("text", "")]
    return "\n".join(parts)


def names_a_target(job: dict) -> bool:
    for step in steps_of(job):
        for line in code_lines(step.get("run") or ""):
            for segment in SEGMENT_RE.split(line):
                if TEST_RE.search(segment) and not INSTALL_RE.search(segment) and not TARGET_RE.search(segment):
                    return False
    return any(has_test_command(s.get("run") or "") for s in steps_of(job))


def triggers(data: dict):
    on = data.get("on", data.get(True))
    if isinstance(on, str):
        return {on: {}}
    if isinstance(on, list):
        return {str(item): {} for item in on}
    if isinstance(on, dict):
        return {str(k): (v if isinstance(v, dict) else {}) for k, v in on.items()}
    return {}


def needs_of(job: dict) -> list[str]:
    needs = job.get("needs")
    if isinstance(needs, str):
        return [needs]
    return [str(n) for n in needs] if isinstance(needs, list) else []


def concurrency_ok(block, target_only: bool = False) -> bool:
    if not isinstance(block, dict):
        return False
    group, cancel = str(block.get("group", "")), block.get("cancel-in-progress")
    if isinstance(cancel, str) and re.search(r"event_name\s*!=\s*['\"]pull_request", cancel):
        cancels = False  # false for the very event it is meant for
    else:
        cancels = cancel is True or (isinstance(cancel, str) and ("pull_request" in cancel or cancel.strip().lower() == "true"))
    per_run = bool(RUN_ID_RE.search(group)) and "pull_request" not in group  # an unconditional run id makes every run its own group
    return cancels and not per_run and bool((TARGET_GROUP_RE if target_only else GROUP_RE).search(group))


def lint_text(name: str, text: str, budget: float = DEFAULT_BUDGET, local: dict | None = None, force_pr: bool = False) -> list[Finding]:
    if yaml is None:
        return [Finding("unreadable", name, "PyYAML is not installed, so the workflow cannot be read")]
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return [Finding("unreadable", name, f"not valid YAML: {str(exc).splitlines()[0]}")]
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), dict) or not data["jobs"]:
        return [Finding("unreadable", name, "no `jobs:` mapping")]
    trig = triggers(data)
    pr_events = [e for e in PR_EVENTS if e in trig] or (["pull_request"] if force_pr else [])
    if not pr_events:
        return []
    top_env = data.get("env") if isinstance(data.get("env"), dict) else None
    jobs = {str(k): (v if isinstance(v, dict) else {}) for k, v in data["jobs"].items()}
    test_jobs = {k for k, v in jobs.items() if is_test_job(v, local, top_env)}
    if not test_jobs:
        return []
    found: list[Finding] = []
    path_filter = all(isinstance(trig.get(e), dict) and ("paths" in trig[e] or "paths-ignore" in trig[e]) for e in pr_events)
    for key in sorted(test_jobs):
        job = jobs[key]
        where = f"{name} job {key}"
        ancestors, todo = set(), list(needs_of(job))
        while todo:
            parent = todo.pop()
            if parent in ancestors or parent not in jobs:
                continue
            ancestors.add(parent)
            todo += needs_of(jobs[parent])
        chained = sorted(ancestors & test_jobs)
        if chained:
            reason = reason_of(job, "TEST_SPEED_CHAIN_REASON", top_env)
            if len(reason) < 20 or len(reason.split()) < 3:
                found.append(Finding("chain", where, f"test stage runs after the test stage {', '.join(chained)} as a separate runner job; {FIX['chain']}"))
        if not (path_filter or SELECT_RE.search(selector_text(job, local)) or names_a_target(job)):
            reason = reason_of(job, "TEST_SPEED_FULL_SUITE_REASON", top_env)
            measured = MEASURED_RE.search(reason)
            if not (measured and 0 < float(measured.group(1)) <= budget):
                why = "its recorded reason has no `measured N min` (0 < N <= budget)" if reason else "it has no selector"
                found.append(Finding("full-suite", where, f"runs every test on every pull-request push: {why}; {FIX['full-suite']}"))
    target_only = "pull_request_target" in pr_events and "pull_request" not in pr_events
    top = data.get("concurrency")
    if not (concurrency_ok(top, target_only) or all(concurrency_ok(jobs[k].get("concurrency"), target_only) for k in jobs)):
        found.append(Finding("concurrency", name, f"a newer push does not cancel the older run; {FIX['concurrency']}"))
    return found


def local_index(files: dict[str, str], actions: dict[str, str] | None) -> dict:
    """For each local reusable workflow and composite action: does it run tests, and the text a selector could live in.
    Computed to a fixed point, so a reference to something that sorts later, or an action inside an action, resolves the same as any other."""
    index: dict[str, dict] = {}
    if yaml is None:
        return index
    parsed: dict[str, tuple[str, dict, dict | None]] = {}
    for name, text in (actions or {}).items():
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        steps = ((data or {}).get("runs") or {}).get("steps") if isinstance(data, dict) else None
        parsed[name] = ("action", {"steps": steps if isinstance(steps, list) else []}, None)
    for name, text in files.items():
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        jobs = data.get("jobs") if isinstance(data, dict) else None
        if isinstance(jobs, dict):
            top_env = data.get("env") if isinstance(data.get("env"), dict) else None
            parsed[f"./.github/workflows/{Path(name).name}"] = ("workflow", {k: v for k, v in jobs.items() if isinstance(v, dict)}, top_env)
    for _ in range(len(parsed) + 1):
        before = json.dumps(index, sort_keys=True)
        for key, (kind, body, top_env) in parsed.items():
            jobs = [body] if kind == "action" else list(body.values())
            index[key] = {"test": any(is_test_job(j, index, top_env) for j in jobs), "text": "\n".join(selector_text(j, index) for j in jobs)}
        if json.dumps(index, sort_keys=True) == before:
            break
    return index


def runs_on_pull_requests(text: str) -> bool:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    return isinstance(data, dict) and any(e in triggers(data) for e in PR_EVENTS)


def concurrency_groups(text: str) -> tuple[str | None, list[str]]:
    """The workflow-level concurrency group and the job-level ones, when the workflow runs on pull requests."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None, []
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), dict) or not any(e in triggers(data) for e in PR_EVENTS):
        return None, []
    top = data.get("concurrency")
    top_group = str(top.get("group", "")) if isinstance(top, dict) else None
    jobs = [str(j["concurrency"].get("group", "")) for j in data["jobs"].values() if isinstance(j, dict) and isinstance(j.get("concurrency"), dict)]
    return top_group, jobs


def called_workflows(files: dict[str, str]) -> set[str]:
    """Names of the local reusable workflows that pull-request workflows call, directly or through other reusable workflows."""
    by_ref = {f"./.github/workflows/{Path(n).name}": n for n in files}
    graph: dict[str, set[str]] = {}
    roots = []
    for name, text in files.items():
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict) or not isinstance(data.get("jobs"), dict):
            continue
        graph[name] = {by_ref[j["uses"]] for j in data["jobs"].values() if isinstance(j, dict) and isinstance(j.get("uses"), str) and j["uses"] in by_ref}
        if any(e in triggers(data) for e in PR_EVENTS):
            roots.append(name)
    seen: set[str] = set()
    todo = list(roots)
    while todo:
        for callee in graph.get(todo.pop(), ()):
            if callee not in seen:
                seen.add(callee)
                todo.append(callee)
    return seen


def lint_files(files: dict[str, str], budget: float = DEFAULT_BUDGET, actions: dict[str, str] | None = None) -> list[Finding]:
    local = local_index(files, actions)
    found: list[Finding] = []
    for name, text in files.items():
        found += lint_text(name, text, budget, local)
    for name in sorted(called_workflows(files)):  # a reusable workflow that chains its own test jobs chains them for every caller
        found += [f for f in lint_text(name, files[name], budget, local, force_pr=True) if f.rule == "chain" and f not in found]
    groups: dict[str, list[str]] = {}
    for name, text in files.items():
        top_group, job_groups = concurrency_groups(text)
        if top_group and "github.workflow" not in top_group:
            groups.setdefault(top_group, []).append(name)
        for group in {g for g in job_groups if job_groups.count(g) > 1 and "github.job" not in g}:
            found.append(Finding("concurrency", name, f"two jobs share the cancelling group {group!r}, so they cancel each other; put `github.job` in the group"))
    for group, names in groups.items():
        if len(names) > 1:  # GitHub shares a group across the repository: these workflows cancel one another, and a required gate with it
            found += [Finding("concurrency", n, f"the group {group!r} is also used by {', '.join(m for m in names if m != n)}; put `github.workflow` in it") for n in names]
    return found


def positive_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def load_config(root: Path) -> dict:
    cfg = {"budget_min": DEFAULT_BUDGET, "workflows": {}, "wall_blocking": False}
    path = Path(root) / CONFIG_NAME
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            if not isinstance(raw, dict):
                raise ValueError("not an object")
            budget = raw.get("budget_min", DEFAULT_BUDGET)
            if not positive_number(budget):
                raise ValueError("budget_min must be a positive finite number")
            if budget > DEFAULT_BUDGET and not (len(str(raw.get("budget_reason", ""))) >= 20 and "measured" in str(raw.get("budget_reason", "")).lower()):
                raise ValueError(f"budget_min above {DEFAULT_BUDGET} needs a budget_reason that says what was measured")
            cfg["budget_min"] = float(budget)
            if not isinstance(raw.get("wall_blocking", False), bool):
                raise ValueError("wall_blocking must be true or false")
            cfg["wall_blocking"] = raw.get("wall_blocking", False)
            workflows = raw.get("workflows") or {}
            if not isinstance(workflows, dict):
                raise ValueError("workflows must be an object")
            cfg["workflows"] = dict(workflows)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
    return cfg


def execution_minutes(row: dict) -> float | None:
    starts, ends = [], []
    for job in row.get("jobs") or []:
        if job.get("conclusion") == "skipped":
            continue
        s, e = epoch(job.get("started_at")), epoch(job.get("completed_at"))
        if s is not None and e is not None:
            starts.append(s)
            ends.append(e)
    return round((max(ends) - min(starts)) / 60, 1) if starts else None


def budget_for(name: str, budget: float, overrides: dict) -> float:
    entry = overrides.get(name)
    if isinstance(entry, dict) and "measured" in str(entry.get("reason", "")).lower() and positive_number(entry.get("budget_min")):
        return float(entry["budget_min"])
    return float(budget)


def row_key(row: dict) -> str:
    return str(row.get("path") or row.get("name"))


def measured_rows(rows: list[dict]) -> list[dict]:
    """The latest finished, not cancelled or skipped, pull-request run of each workflow file."""
    latest: dict[str, tuple] = {}
    for row in rows:
        if row.get("event") not in PR_EVENTS or row.get("conclusion") in (None, "cancelled", "skipped", "stale"):
            continue
        key = row_key(row)
        rank = (epoch(row.get("created_at")) or 0, row.get("id") or 0)
        if key not in latest or rank > latest[key][0]:
            latest[key] = (rank, row)
    return [pair[1] for pair in latest.values()]


def wall_findings(rows: list[dict], budget: float, overrides: dict) -> list[Finding]:
    found = []
    for row in measured_rows(rows):
        minutes = execution_minutes(row)
        limit = budget_for(str(row.get("name")), budget, overrides)
        if minutes is not None and minutes > limit:
            found.append(Finding("wall-time", str(row.get("name")),
                                 f"{row.get('name')} ran {minutes} min on a pull request (run {row.get('id')}), over its {limit:g} min budget; "
                                 "select fewer tests, merge stages or record a measured per-workflow budget in .test-speed.json"))
    return found


def http_get(url: str, token: str):
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}", "User-Agent": "test-speed"})
    with urllib.request.urlopen(request, timeout=20) as reply:
        return json.load(reply)


def fetch_runs(repo: str, branch: str, token: str, current_run: str | None = None, pr_number: int | None = None, head_repo: str | None = None,
               present: set[str] | None = None, get=http_get) -> list[dict]:
    """Finished runs of this pull request. A run is tied to it by its pull-request list; GitHub leaves that list empty for some runs, so an empty
    one counts when it ran on this branch of the pull request's head repository. Only the latest run of each workflow file still in the tree
    has its jobs read (one request each); a request that fails drops that run, never the rest."""
    query = urllib.parse.urlencode({"branch": branch, "status": "completed", "per_page": 100})
    latest: dict[str, dict] = {}
    for run in get(f"https://api.github.com/repos/{repo}/actions/runs?{query}", token).get("workflow_runs", []):
        if str(run["id"]) == str(current_run) or run.get("conclusion") in ("cancelled", "skipped") or run.get("event") not in PR_EVENTS:
            continue
        path = str(run.get("path") or "").split("@")[0]
        if present is not None and path not in present:
            continue
        if pr_number is not None:
            numbers = [p.get("number") for p in run.get("pull_requests") or []]
            if numbers:
                if int(pr_number) not in numbers:
                    continue
            elif run.get("head_branch") != branch or (run.get("head_repository") or {}).get("full_name") != (head_repo or repo):
                continue
        rank = (epoch(run.get("created_at")) or 0, run["id"])
        if path not in latest or rank > latest[path][0]:
            latest[path] = (rank, run, path)
    rows = []
    for _, run, path in latest.values():
        try:
            jobs = get(f"https://api.github.com/repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100&filter=latest", token).get("jobs", [])
        except Exception as exc:
            print(f"test-speed: jobs of run {run['id']} not read ({type(exc).__name__})", file=sys.stderr)
            continue
        rows.append({"id": run["id"], "name": run.get("name"), "path": path, "event": run.get("event"), "head_sha": run.get("head_sha"),
                     "conclusion": run.get("conclusion"), "created_at": run.get("created_at"),
                     "jobs": [{k: j.get(k) for k in ("started_at", "completed_at", "conclusion")} for j in jobs]})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--no-wall", action="store_true", help="lint only; skip the time budget")
    ap.add_argument("--runs-json", help="finished run rows from a file instead of GitHub (tests, replays)")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    ap.add_argument("--branch", default=os.environ.get("GITHUB_HEAD_REF"))
    ap.add_argument("--pr-number", type=int, default=int(os.environ["PR_NUMBER"]) if os.environ.get("PR_NUMBER", "").isdigit() else None)
    ap.add_argument("--head-repo", default=os.environ.get("HEAD_REPO"))
    ap.add_argument("--head-sha", default=os.environ.get("HEAD_SHA"))
    ap.add_argument("--current-run", default=os.environ.get("GITHUB_RUN_ID"))
    args = ap.parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"test-speed: cannot read {root}", file=sys.stderr)
        return 2
    github, workflows, actions_dir = root / ".github", root / ".github" / "workflows", root / ".github" / "actions"
    files = sorted(p for pattern in ("*.yml", "*.yaml") for p in workflows.glob(pattern))
    action_dirs = sorted(actions_dir.glob("*")) if actions_dir.is_dir() else []
    action_files = [q for d in action_dirs for q in (d / "action.yml", d / "action.yaml") if q.exists() or q.is_symlink()]
    reads = [github, workflows, actions_dir, root / CONFIG_NAME, *files, *action_dirs, *action_files]
    real_root = os.path.realpath(root)
    if any(q.is_symlink() or not (os.path.realpath(q) + os.sep).startswith(real_root + os.sep) for q in reads):
        print("test-speed: refusing a path that is a symlink or resolves outside the checkout (workflows, actions or config): the check reads only files inside the tree", file=sys.stderr)
        return 2
    try:  # the guard above has already judged every path; nothing is opened before it
        cfg = load_config(root)
    except ValueError as exc:
        print(f"test-speed: unreadable config {exc}", file=sys.stderr)
        return 2
    budget = cfg["budget_min"]
    actions = {f"./.github/actions/{q.parent.name}": q.read_text() for q in action_files}
    found: list[Finding] = []
    lines: list[str] = []
    found += lint_files({p.relative_to(root).as_posix(): p.read_text() for p in files}, budget, actions)
    if not files:
        lines.append("test-speed: no workflows, nothing to check")
    if not args.no_wall:
        rows = None
        if args.runs_json:
            rows = json.loads(Path(args.runs_json).read_text())
        elif args.repo and args.branch and os.environ.get("GITHUB_TOKEN"):
            try:
                rows = fetch_runs(args.repo, args.branch, os.environ["GITHUB_TOKEN"], args.current_run, args.pr_number, args.head_repo,
                                  {p.relative_to(root).as_posix() for p in files})
            except Exception as exc:  # the lint above still decides; a GitHub outage must not block every pull request
                lines.append(f"test-speed: wall time not measured ({type(exc).__name__}); the alarm still watches it")
        if rows is not None:
            present = {p.relative_to(root).as_posix() for p in files}
            rows = [r for r in rows if not r.get("path") or r["path"] in present]  # a workflow this pull request removed is not judged
            measured = measured_rows(rows)
            for row in measured:
                minutes = execution_minutes(row)
                old = f" (older commit {str(row.get('head_sha'))[:7]})" if args.head_sha and row.get("head_sha") != args.head_sha else ""
                lines.append(f"test-speed: wall {row.get('name')} run {row.get('id')} {minutes} min{old} (budget {budget_for(str(row.get('name')), budget, cfg['workflows']):g})")
            have = {r.get("path") for r in measured}
            for p in files:
                rel = p.relative_to(root).as_posix()
                if rel not in have and runs_on_pull_requests(p.read_text()):
                    lines.append(f"test-speed: wall pending {rel}: no finished pull-request run of this pull request yet; the next push and the CI speed alarm judge it")
            walls = wall_findings(rows, budget, cfg["workflows"])
            if cfg.get("wall_blocking"):
                found += walls
            else:  # a flag with its evidence: the verdict on a finished older run cannot bind a newer one, so the alarm's median is the hard stop
                lines += [f"::warning::test-speed: {w.rule}: {w.where}: {w.message}" for w in walls]
    for f in found:
        lines.append(f"test-speed: {f.rule}: {f.where}: {f.message}")
    if not found and files:
        lines.append(f"test-speed: ok ({len(files)} workflows, budget {budget:g} min)")
    print("\n".join(lines))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:  # the verdict is the exit code and stdout above; a read-only or missing summary file (the runner's sandbox) must not change it
            with open(summary, "a") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError as exc:
            print(f"test-speed: step summary not written ({type(exc).__name__})", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
