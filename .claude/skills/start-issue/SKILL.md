---
name: start-issue
description: Pick the next Linear issue (or take the one named), make or find its GitHub twin, claim it, and create the worktree. Use when asked what to work on next, or to start ALE-nn or a GitHub issue.
---

# Start an issue

Linear holds the backlog and decides what is next. GitHub is where the
branch, PR and review live. This skill goes from a Linear issue to a claimed
GitHub issue, a plan and a worktree. AGENTS.md, "Change, GitHub, and Linear
hygiene", says why (issue #705, ALE-311).

## 0. Pick the issue, if none was named

- `list_issues(project="Constellations", state="Todo")`. Drop every issue with
  open children: a parent is never picked, so take its next child.
- Order by the target date of the issue's milestone (none sorts last), then by
  priority (Urgent, High, Medium, Low, None).
- Count the project's leaf issues In Progress. At five or more, propose
  finishing one of those instead, unless the user says otherwise.
- Propose the top one or two, with a line each on why, and let the user
  choose. When nothing is in Todo, say so and offer the best Backlog
  candidates rather than picking one silently.
- Given a GitHub number instead, find its Linear twin from the body's
  `Synced from Linear [ALE-nn]` footer. If it has none, file it in Linear
  first (see AGENTS.md), then continue with that `ALE-nn`.

## 1. Load the Linear issue

`get_issue` with `ALE-nn`: description, status, parent, milestone and
attachments. Summarize the requirement and the done-when. If it is unplaced
(no parent, or still in Triage), place it first.

## 2. Find or create its GitHub twin

```bash
python scripts/linear_twin.py ALE-nn
```

- Exit 0 prints the twin's number.
- Exit 1 means none. Write the body to a file: first line
  `Theme: ALE-<parent> — <parent title>`, then the Linear description, with
  GitHub references written bare (`#701`) since it now lives on GitHub. Then
  `python scripts/linear_twin.py ALE-nn --create --title "<title>" --body-file <file>`.
  It adds the footer and refuses to make a second twin.
- Exit 2 means it could not check. Create nothing.
- Never search GitHub for `ALE-nn` instead: search matches substrings.
- Attach the twin in Linear: `save_issue` with
  `links: [{url: "<issue url>", title: "GH #N"}]`.

## 3. Check for a claim, then claim

```bash
python scripts/check_issue_claim.py <N>
```

It looks at assignees, claim comments, open PRs (drafts too) naming the issue,
merged PRs that already close it, and remote branches, local branches and
worktrees named `feat/<N>-…`. Exit 0 is clear. **Exit 1 means stop and ask
the user** rather than open a second implementation. Exit 2 is "not
verified", never "clear".

When clear, choose the branch name now and claim:

```bash
python scripts/check_issue_claim.py <N> --claim --branch feat/<N>-<slug>
```

(`fix/` for a bug fix, `docs/` for documentation.) Claim *before* creating the
branch: the check reports a local branch of that name as someone's work.

## 4. Move the Linear issue to In Progress

`save_issue` with `id: "ALE-nn"`, `state: "In Progress"`. Never a parent.

## 5. Create a worktree from origin, not a branch in place

```bash
git fetch origin
git worktree add -b feat/<N>-<slug> <scratchpad>/wt-<N> origin/master
uv sync --all-extras --dev --locked --python 3.12
```

From `origin/master` after a fetch, explicitly: a bare `git checkout -b`
inherits whatever `HEAD` is, and another session can switch the primary
checkout's `HEAD` under you. Not Linear's `gitBranchName`: this repo's
tooling keys off the GitHub number, and the PR body's `Fixes ALE-nn` is what
links Linear.

## 6. Plan, implement, validate

Scope questions before code. A short ordered plan. Then the validation set in
AGENTS.md, "Development workflow".

## 7. Before opening the PR

```bash
python scripts/check_issue_claim.py <N> --mine feat/<N>-<slug>
```

That ignores your own branch, PR and claim, and reports anything that appeared
while you worked. A hit means ask before pushing a competing PR. Then open
the PR with `Closes #N`, `Fixes ALE-nn` and `Theme: ALE-<parent>` (see the
`linear-sync` skill), and attach the PR in Linear.

## Notes

- File new work in Linear, never with `gh issue create`. The GitHub issue is
  made here, when work starts.
- When a PR merges and the issue stays open for another piece, release the
  claim (`--release`) and claim again with the next branch.
