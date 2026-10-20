"""Is anyone already working GitHub issue N? Check before you start, claim when clear.

    python scripts/check_issue_claim.py 705
    python scripts/check_issue_claim.py 705 --claim --branch feat/705-linear-first-triage
    python scripts/check_issue_claim.py 705 --mine feat/705-linear-first-triage  # before PR
    python scripts/check_issue_claim.py 705 --release   # drop a claim nobody is working

Claim before you create the branch: `--branch` names a branch you intend to
use, not one you own yet, so an existing branch, PR or claim under that same
name is still reported. Only `--mine`, once the claim is yours, excludes
your own branch, PR and claim.

Exit 0: nothing found (and, with --claim, the claim was posted).
Exit 1: someone may already have it. Stop and ask the user; do not open a
        second implementation.
Exit 2: the check itself could not run (gh or git failed). Treat as "not
        verified", never as clear.

--release removes the issue's claim comments and unassigns you. A claim has no
other way out. It refuses (exit 1) while any branch a claim names is live -- an
open PR; pushed with no PR yet, or pushed since its last PR; or never pushed but
still a local branch or worktree -- and it refuses a claim that names no branch
it can read, because deleting another session's live claim is how a second
implementation starts. --force overrides that, for work you know is dead.

Why this exists: every session here runs as the same GitHub account, so
"assigned to me" cannot tell this session from another one. The claim comment
names the branch, which is what lets the next session tell whose claim it is.
In astrolabe two sessions built one issue independently for eleven hours, in
different architectures (astrolabe#634 and astrolabe#637). Ported from
astrolabe (astrolabe#1181) for issue #705, with stel's `feat/<N>-<slug>`
branch names.

Looks in six places: the issue's assignees, claim comments on it, open PRs
(drafts included) naming it in title, body or head branch, recently merged PRs
that close it, remote branches, and local branches and worktrees. Standard
library only, so it runs from any checkout with plain `python`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

CLAIM_MARKER = "<!-- issue-claim -->"
MERGED_LOOKBACK = 30  # merged PRs to scan for a `Closes #N` that already shipped


class CheckError(RuntimeError):
    pass


def run(args: list[str]) -> str:
    try:
        done = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", check=False)
    except FileNotFoundError as exc:
        raise CheckError(f"{args[0]} is not installed: {exc}") from exc
    if done.returncode != 0:
        raise CheckError(f"`{' '.join(args)}` failed: {done.stderr.strip()}")
    return done.stdout


def gh_json(args: list[str]) -> Any:
    return json.loads(run(["gh", *args]) or "null")


def references(text: str | None, number: int) -> bool:
    """`#705`, `issue-705`, `issues/705` -- but not `#7050` or `issue-70`."""
    if not text:
        return False
    pattern = rf"(#|issue[-_ ]|issues/){number}(?!\d)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def branch_names_issue(branch: str, number: int) -> bool:
    """`feat/705-slug`, `origin/fix/705-slug`, `feat/issue-705-slug`.

    The number must open a path segment, so `release/v0.23.0` is not issue 23
    and `feat/7050-x` is not issue 705.
    """
    return re.search(rf"(?:^|/)(?:issue-)?{number}(?=-|$)", branch) is not None


def check(number: int, mine: str | None) -> list[str]:
    """Everything that says someone may already be working `number`.

    `mine` is this session's own branch, passed only on a `--mine` re-check:
    its branch, PR and claim are ignored, and so is this account's assignee
    entry when the latest claim is this branch's. Another assignee is always
    reported.
    """
    findings: list[str] = []

    issue = gh_json(["issue", "view", str(number), "--json", "state,title,assignees,comments"])
    if issue["state"] != "OPEN":
        findings.append(f"issue #{number} is {issue['state']}")

    claims = [c for c in issue["comments"] if CLAIM_MARKER in (c.get("body") or "")]
    latest_claim = claims[-1] if claims else None
    claimed_by_me = bool(
        latest_claim and mine and f"`{mine}`" in (latest_claim.get("body") or "")
    )
    for claim_comment in claims:
        body = claim_comment.get("body") or ""
        if mine and f"`{mine}`" in body:
            continue
        first_line = body.replace(CLAIM_MARKER, "").strip()
        findings.append(
            f"claim comment {claim_comment.get('createdAt', '')[:16]}: {first_line[:200]}"
        )

    assignees = [a["login"] for a in issue["assignees"]]
    if claimed_by_me:
        me = run(["gh", "api", "user", "--jq", ".login"]).strip()
        assignees = [login for login in assignees if login != me]
    if assignees:
        findings.append(
            f"assigned to {', '.join(assignees)} (every session shares one "
            "GitHub account, so this may be another session of yours)"
        )

    findings.extend(_open_pr_findings(number, mine))
    findings.extend(_merged_pr_findings(number))
    findings.extend(_branch_findings(number, mine))
    # A local branch with a worktree is reported twice; keep the first wording.
    return list(dict.fromkeys(findings))


def _open_pr_findings(number: int, mine: str | None) -> list[str]:
    open_prs = gh_json(
        [
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "300",
            "--json",
            "number,title,headRefName,body,isDraft,url",
        ]
    )
    findings: list[str] = []
    for pr in open_prs:
        if mine and pr["headRefName"] == mine:
            continue
        if (
            references(pr["title"], number)
            or references(pr["body"], number)
            or branch_names_issue(pr["headRefName"], number)
        ):
            draft = " (draft)" if pr["isDraft"] else ""
            findings.append(
                f"open PR #{pr['number']}{draft} on {pr['headRefName']}: "
                f"{pr['title']} {pr['url']}"
            )
    return findings


def _merged_pr_findings(number: int) -> list[str]:
    merged = gh_json(
        [
            "pr",
            "list",
            "--state",
            "merged",
            "--limit",
            str(MERGED_LOOKBACK),
            "--json",
            "number,title,closingIssuesReferences,mergedAt",
        ]
    )
    findings: list[str] = []
    for pr in merged:
        closes = {ref.get("number") for ref in pr.get("closingIssuesReferences") or []}
        if number in closes:
            findings.append(
                f"merged PR #{pr['number']} ({pr['mergedAt'][:10]}) closes "
                f"#{number}: {pr['title']} -- the work may already have shipped"
            )
    return findings


def _branch_findings(number: int, mine: str | None) -> list[str]:
    findings: list[str] = []
    run(["git", "fetch", "--quiet", "--prune", "origin"])
    for line in run(["git", "branch", "-r", "--format=%(refname:short)"]).splitlines():
        branch = line.strip()
        if branch_names_issue(branch, number) and branch.removeprefix("origin/") != mine:
            findings.append(f"remote branch {branch}")

    for line in run(["git", "branch", "--format=%(refname:short)"]).splitlines():
        branch = line.strip()
        if branch_names_issue(branch, number) and branch != mine:
            findings.append(f"local branch {branch}")

    worktree = None
    for line in run(["git", "worktree", "list", "--porcelain"]).splitlines():
        if line.startswith("worktree "):
            worktree = line.removeprefix("worktree ")
            continue
        if not line.startswith("branch "):
            continue
        branch = line.removeprefix("branch refs/heads/")
        if branch_names_issue(branch, number) and branch != mine:
            findings.append(f"worktree {worktree} on {branch}")
    return findings


class LostClaim(RuntimeError):
    """Another session's claim landed first; ours was withdrawn."""


def _claim_comments(number: int) -> list[dict[str, Any]]:
    """Every claim comment on the issue, oldest first (the API's order)."""
    out = run(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{{owner}}/{{repo}}/issues/{number}/comments",
            "--jq",
            f'.[] | select(.body | contains("{CLAIM_MARKER}")) | {{id, created_at, body}}',
        ]
    )
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def _delete_comment(comment_id: int) -> None:
    run(["gh", "api", "-X", "DELETE", f"repos/{{owner}}/{{repo}}/issues/comments/{comment_id}"])


def claim(number: int, branch: str) -> None:
    """Post the claim, confirm it is the first one, then assign.

    Two sessions can both pass check() before either posts, so passing it
    is not ownership. Each run's comment carries a token only it knows
    (the branch name can be the same for both); after posting, the
    earliest claim comment on the issue wins. A run that loses deletes its
    own comment and raises LostClaim. Assignment comes last, and a failure
    there deletes the comment, so nothing is ever left half-claimed.
    """
    token = uuid.uuid4().hex
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    body = (
        f"{CLAIM_MARKER}<!-- claim-token: {token} -->\n"
        f"Claimed by an agent session on branch `{branch}`, {stamp}. Another "
        f"session: run `python scripts/check_issue_claim.py {number}` "
        "and ask before starting a second implementation."
    )
    comments_path = f"repos/{{owner}}/{{repo}}/issues/{number}/comments"
    posted = json.loads(run(["gh", "api", comments_path, "-f", f"body={body}"]))
    mine_id = posted["id"]
    try:
        claims = _claim_comments(number)
        if not claims or token not in claims[0]["body"]:
            winner = claims[0]["body"] if claims else "(none found)"
            _delete_comment(mine_id)
            first_line = winner.split("\n", 1)[-1].strip()
            raise LostClaim(f"another claim landed first: {first_line[:200]}")
        run(["gh", "issue", "edit", str(number), "--add-assignee", "@me"])
    except CheckError:
        _delete_comment(mine_id)
        raise


# "on branch `x`" is what --claim writes; "for branch `x`" is how a claim
# written by hand reads. Both name the branch.
_CLAIM_BRANCH = re.compile(r"branch\s+`([^`]+)`")


def claimed_branches(body: str) -> list[str]:
    """Every branch a claim comment names. A claim written by hand can name two,
    and one live branch keeps the claim live."""
    return _CLAIM_BRANCH.findall(body or "")


def _exists_locally(branch: str) -> bool:
    """A local branch, or a worktree on it: check() counts both as someone's work."""
    if run(["git", "branch", "--list", branch]).strip():
        return True
    return any(
        line.removeprefix("branch refs/heads/") == branch
        for line in run(["git", "worktree", "list", "--porcelain"]).splitlines()
        if line.startswith("branch ")
    )


def live_work(branch: str) -> str | None:
    """Why `branch` still looks like someone's work, or None when it does not.

    Live: an open PR; pushed with no PR yet (a session between its first push and
    `gh pr create`); pushed since its last PR (the remote tip is not that PR's
    head); or never pushed but present as a local branch or worktree (a session
    between claiming and its first push). Dead: its PR is merged or closed and
    nothing newer is on origin, or it exists nowhere.
    """
    prs = gh_json(
        ["pr", "list", "--head", branch, "--state", "all", "--json", "number,state,headRefOid"]
    )
    open_prs = [pr for pr in prs if pr["state"] == "OPEN"]
    if open_prs:
        return f"it has an open PR (#{open_prs[0]['number']})"
    remote = run(["git", "ls-remote", "--heads", "origin", branch]).split()
    if remote:
        if not prs:
            return "it is pushed and has no PR yet"
        if remote[0] not in {pr["headRefOid"] for pr in prs}:
            return "it has been pushed since its last PR"
        return None
    if not prs and _exists_locally(branch):
        return "it exists locally and was never pushed"
    return None


def release(number: int, force: bool) -> int:
    """Delete the issue's claim comments and unassign this account.

    Every claim comment goes, so every one must be dead: a claim whose branch
    cannot be read is refused (it may be live and just worded by hand), and so is
    any claim whose branch looks live, whichever comment names it. `claim()`
    treats the earliest claim as the owner, so the latest alone proves nothing.

    Returns the exit code: 0 released (or nothing to release), 1 refused.
    """
    claims = _claim_comments(number)
    assigned = gh_json(["issue", "view", str(number), "--json", "assignees"])["assignees"]
    if not claims and not assigned:
        print(f"#{number}: no claim and no assignee. Nothing to release.")
        return 0
    per_claim = [claimed_branches(c["body"]) for c in claims]
    branches = list(dict.fromkeys(b for names in per_claim for b in names))
    if not force:
        refusal = _release_refusal(number, per_claim, branches)
        if refusal is not None:
            print(refusal)
            return 1
    for comment in claims:
        _delete_comment(comment["id"])
    if assigned:
        run(["gh", "issue", "edit", str(number), "--remove-assignee", "@me"])
    named = ", ".join(f"`{b}`" for b in branches)
    print(
        f"#{number}: released {len(claims)} claim(s)"
        f"{' on ' + named if named else ''}"
        f"{' and unassigned you' if assigned else ''}."
    )
    return 0


def _release_refusal(number: int, per_claim: list[list[str]], branches: list[str]) -> str | None:
    if any(not names for names in per_claim):
        return (
            f"#{number}: not released. A claim comment names no branch I can "
            "read, so I cannot tell whether its work is live. Check by hand, "
            "then re-run with --force."
        )
    for branch in branches:
        reason = live_work(branch)
        if reason:
            return (
                f"#{number}: not released. A claim names `{branch}` and "
                f"{reason}. If that work is dead, re-run with --force."
            )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("issue", type=int)
    parser.add_argument(
        "--mine",
        help="this session's own branch: ignore it, its PR and its claim "
        "(use when re-checking just before `gh pr create`)",
    )
    parser.add_argument(
        "--claim",
        action="store_true",
        help="when nothing is found, assign @me and post a claim comment naming --branch",
    )
    parser.add_argument(
        "--branch",
        help="the branch the claim names (with --claim); a candidate, not "
        "yet yours, so anything already under that name is still reported",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="remove this issue's claim comments and unassign it; refuses while "
        "a claimed branch has an open PR, is pushed with no PR yet or since its "
        "last PR, or exists only locally, and when a claim names no readable "
        "branch (use --force once you have checked)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --release, release even a branch that looks live",
    )
    args = parser.parse_args()
    if args.force and not args.release:
        parser.error("--force only applies to --release")
    if args.release and (args.claim or args.mine or args.branch):
        parser.error("--release stands alone: it does not check, claim or re-check")
    if args.release:
        try:
            return release(args.issue, args.force)
        except CheckError as exc:
            print(f"Release failed: {exc}", file=sys.stderr)
            return 2
    if args.claim and args.mine:
        parser.error(
            "--claim and --mine together would hide an existing claim on the "
            "branch you are about to claim; claim first, re-check with --mine later"
        )
    if args.claim and not args.branch:
        parser.error("--claim needs --branch, so the next session can see whose claim it is")

    try:
        findings = check(args.issue, args.mine)
    except CheckError as exc:
        print(f"NOT VERIFIED: {exc}", file=sys.stderr)
        return 2

    if findings:
        print(f"#{args.issue} may already be taken -- stop and ask before working it:")
        for finding in findings:
            print(f"  - {finding}")
        if args.claim:
            print("Not claimed.")
        return 1

    print(f"#{args.issue}: no assignee, claim, PR or branch found.")
    if args.claim:
        try:
            claim(args.issue, args.branch)
        except LostClaim as exc:
            print(f"Not claimed: {exc}. Stop and ask before working it.")
            return 1
        except CheckError as exc:
            print(f"Claim failed: {exc}", file=sys.stderr)
            return 2
        print(f"Claimed #{args.issue} for `{args.branch}`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
