---
name: linear-sync
description: File work in Linear and keep a Linear issue, its GitHub twin and its PRs linked. Use when filing an issue, opening a PR, or after one merges — and when asked where an issue belongs, or why Linear and GitHub disagree about what is open.
---

# Linear and GitHub, kept in step

**Linear holds the backlog; GitHub holds the work** (AGENTS.md, "Change,
GitHub, and Linear hygiene"; issue #705, ALE-311). A leaf Linear issue has
exactly one GitHub twin, made when work starts. A theme is a Linear parent
issue with leaf children; it never has a twin of its own.

- Team prefix `ALE`. Projects: **Constellations** (this repo, `stel`) and
  **Astrolabe** (the downstream data project).
- To start work, use the `start-issue` skill. This one covers filing, the PR
  and the merge.

## Filing an issue

1. Find its theme. Query, never a list written down anywhere:
   `list_issues(project="Constellations", state="Backlog|Todo")`, top-level
   issues only. Match on what the work is *about*, not which files it touches.
   If nothing fits, say so and ask: a new theme is a decision about the
   project's shape.
2. `save_issue` with `team: "ALE"`, `project`, `parentId` (the theme),
   `milestone` (the theme's, unless the issue belongs to a different phase),
   `priority`, and `state` Todo or Backlog. Give it a problem statement and a
   done-when.
3. Write GitHub references with their repo in Linear text
   (`constellations#701`, `astrolabe#1178`): Linear links a bare `#701` to
   whatever has that number in another repo.
4. Do not create the GitHub issue. `start-issue` does, when work starts.

An issue that already exists on GitHub (filed from outside, or older) gets its
Linear issue now: file it as above, append
`Synced from Linear [ALE-nn](https://linear.app/alex-noonan/issue/ALE-nn)` to
the GitHub body, and attach the GitHub link in Linear (`save_issue` `links`).

## Opening a PR

In the Tracking section the template provides:

```
- Closes #705
- Fixes ALE-311
- Theme: ALE-48
```

- `Closes #N` only when the PR satisfies the GitHub issue's done-when;
  `Part of #N` otherwise.
- `Fixes ALE-nn` names the **leaf** issue, so Linear's GitHub app links the PR
  and moves the issue to Done on merge. `Part of ALE-nn` links without closing.
- **Never a closing keyword on a theme.** `Fixes`, `Closes` or `Resolves`
  against a parent would complete it on merge and hide every child still
  open. Write the parent as `Theme: ALE-nn`.
- Attach the PR in Linear as well (`save_issue` `links`), so the link exists
  even if the app misses it.

## After a merge

- Check that the leaf issue went to Done. Done means shipped and verified.
- If something is still owed (a release, a check against astrolabe's prod),
  move it to **In Review** with a comment saying what is owed and how it will
  be checked. Move it to Done once that is recorded. In Review never means
  "PR open, awaiting review"; that stays In Progress.
- A theme closes when a human decides it is done, not when its last PR lands.
  If its children are all closed, say so in a comment on the theme, naming the
  evidence, and leave the decision to them.

## When the trackers disagree

- A Linear issue still open after its GitHub issue and every PR closed:
  suggest Done or Canceled.
- A GitHub issue with no `Synced from Linear` footer: it has no Linear issue
  yet. File one.
- A parent In Progress: move it to Todo or Backlog; its state is its children's.
- More than five leaf issues In Progress: finish one before starting another.
