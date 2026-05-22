"""Rewrite all commits on current branch to add a 'Files:' line.

Run from repo root:  uv run python scripts/rewrite_commits.py
"""
import json
import os
import subprocess
import sys


def run(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True, encoding="utf-8").strip()


def run_stdin(cmd: str, stdin: str, extra_env: dict | None = None) -> str:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        cmd, shell=True, input=stdin, text=True, encoding="utf-8",
        capture_output=True, env=env,
    )
    if result.returncode != 0:
        print("ERROR:", result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def file_changes(sha: str) -> str:
    """Return a human-readable 'Files:' value for this commit."""
    raw = run(f"git diff-tree --no-commit-id -r --name-status -M {sha}")
    parts = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        cols = line.split("\t")
        status = cols[0][0]          # first char: M A D R C
        if status == "M":
            parts.append(cols[1])
        elif status == "A":
            parts.append(f"new {cols[1]}")
        elif status == "D":
            parts.append(f"deleted {cols[1]}")
        elif status in ("R", "C"):   # renamed / copied
            parts.append(f"{cols[1]} -> {cols[2]}")
    return ", ".join(parts) if parts else "(none)"


def main() -> None:
    branch = run("git branch --show-current")
    if not branch:
        print("ERROR: not on a named branch")
        sys.exit(1)

    # oldest → newest
    shas = run("git log --reverse --format=%H").splitlines()
    print(f"Rewriting {len(shas)} commits on '{branch}' ...\n")

    new_parent: str | None = None

    for sha in shas:
        old_msg   = run(f"git log -1 --format=%B {sha}")
        tree      = run(f"git log -1 --format=%T {sha}")
        a_name    = run(f"git log -1 --format=%an {sha}")
        a_email   = run(f"git log -1 --format=%ae {sha}")
        a_date    = run(f"git log -1 --format=%aI {sha}")
        c_name    = run(f"git log -1 --format=%cn {sha}")
        c_email   = run(f"git log -1 --format=%ce {sha}")
        c_date    = run(f"git log -1 --format=%cI {sha}")

        files_val = file_changes(sha)
        new_msg   = old_msg.rstrip() + f"\n\nFiles: {files_val}"

        parent_flag = f"-p {new_parent}" if new_parent else ""
        cmd = f"git commit-tree {tree} {parent_flag}"

        new_sha = run_stdin(cmd, new_msg, extra_env={
            "GIT_AUTHOR_NAME":     a_name,
            "GIT_AUTHOR_EMAIL":    a_email,
            "GIT_AUTHOR_DATE":     a_date,
            "GIT_COMMITTER_NAME":  c_name,
            "GIT_COMMITTER_EMAIL": c_email,
            "GIT_COMMITTER_DATE":  c_date,
        })

        subject = new_msg.splitlines()[0]
        print(f"  {sha[:7]} → {new_sha[:7]}  {subject}")
        new_parent = new_sha

    run(f"git update-ref refs/heads/{branch} {new_parent}")
    run(f"git checkout {branch}")   # refresh HEAD pointer
    print(f"\nDone — '{branch}' now at {new_parent[:7]}")


if __name__ == "__main__":
    main()
