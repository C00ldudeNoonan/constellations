"""Find, or create, the GitHub issue for a Linear issue (issue #705, ALE-311).

    python scripts/linear_twin.py ALE-311
    python scripts/linear_twin.py ALE-311 --create --title "..." --body-file body.md

Linear holds the backlog. A GitHub issue is created when work starts, because
that is where the branch, PR and review live. Nothing syncs Linear to GitHub
automatically, so the `start-issue` skill makes the twin, and this script
keeps it from making two. Ported from astrolabe (astrolabe#1181, ALE-228).

A twin is a GitHub issue whose body carries the footer
`Synced from Linear [ALE-nn](url)`. Nothing else counts:

- `Theme: ALE-nn` on the first line names the parent, not a twin.
- A bare `ALE-nn` is a mention, and GitHub search matches substrings. On
  2026-10-13 a search for "ALE-228" returned an unrelated issue in astrolabe.

Issues are listed (`gh issue list --state all`) and matched here rather than
searched, because search lags a just-created issue.

`--create` appends the footer when the body lacks it, files the issue, and
prints its number. It refuses when a twin already exists.

Exit 0: a twin was found, or created with --create. Its number is printed.
Exit 1: no twin, and --create was not given.
Exit 2: the check could not run (gh failed or bad input). Treat this as
        "not verified", never as "none".

Standard library only, so it runs from any checkout with plain `python`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

LINEAR_ISSUE_URL = "https://linear.app/alex-noonan/issue/{identifier}"
LIST_LIMIT = 5000
_IDENTIFIER = re.compile(r"^[A-Z]+-\d+$")


class CheckError(RuntimeError):
    pass


def footer_pattern(identifier: str) -> re.Pattern[str]:
    # The closing bracket keeps ALE-31 from matching ALE-311's footer.
    return re.compile(rf"Synced from Linear \[{re.escape(identifier)}\]")


def footer(identifier: str) -> str:
    url = LINEAR_ISSUE_URL.format(identifier=identifier)
    return f"Synced from Linear [{identifier}]({url})"


def twins(issues: list[dict[str, Any]], identifier: str) -> list[int]:
    """Numbers of the issues whose body carries this identifier's footer."""
    pattern = footer_pattern(identifier)
    return sorted(
        int(issue["number"]) for issue in issues if pattern.search(issue.get("body") or "")
    )


def with_footer(body: str, identifier: str) -> str:
    if footer_pattern(identifier).search(body):
        return body
    return f"{body.rstrip()}\n\n{footer(identifier)}\n"


def run(args: list[str], stdin: str | None = None) -> str:
    try:
        done = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            input=stdin,
            check=False,
        )
    except FileNotFoundError as exc:
        raise CheckError(f"{args[0]} is not installed: {exc}") from exc
    if done.returncode != 0:
        raise CheckError(f"`{' '.join(args)}` failed: {done.stderr.strip()}")
    return done.stdout


def list_issues() -> list[dict[str, Any]]:
    out = run(
        [
            "gh",
            "issue",
            "list",
            "--state",
            "all",
            "--limit",
            str(LIST_LIMIT),
            "--json",
            "number,body",
        ]
    )
    return json.loads(out)


def create(title: str, body: str) -> int:
    out = run(["gh", "issue", "create", "--title", title, "--body-file", "-"], body)
    match = re.search(r"/issues/(\d+)", out)
    if match is None:
        raise CheckError(f"gh issue create printed no issue URL: {out.strip()}")
    return int(match.group(1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find, or create, the GitHub issue for a Linear issue."
    )
    parser.add_argument("identifier", help="Linear identifier, e.g. ALE-311")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--title")
    parser.add_argument("--body-file", type=Path)
    args = parser.parse_args(argv)

    identifier = args.identifier.strip().upper()
    if not _IDENTIFIER.match(identifier):
        print(f"Not a Linear identifier: {args.identifier}", file=sys.stderr)
        return 2
    if args.create and not (args.title and args.body_file):
        print("--create needs --title and --body-file", file=sys.stderr)
        return 2

    try:
        found = twins(list_issues(), identifier)
        if found:
            if len(found) > 1:
                print(
                    f"{identifier} has {len(found)} GitHub twins: "
                    + ", ".join(f"#{n}" for n in found)
                    + ". Close the extras as duplicates.",
                    file=sys.stderr,
                )
            print(found[-1])
            return 0
        if not args.create:
            print(f"{identifier} has no GitHub twin.", file=sys.stderr)
            return 1
        body = with_footer(args.body_file.read_text(encoding="utf-8"), identifier)
        number = create(args.title, body)
    except (CheckError, OSError, json.JSONDecodeError) as exc:
        print(f"Could not check {identifier}: {exc}", file=sys.stderr)
        return 2
    print(number)
    return 0


if __name__ == "__main__":
    sys.exit(main())
