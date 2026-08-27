---
name: linear-sync
description: Keep GitHub issues and PRs linked to their Linear theme, in both directions. Use when filing a GitHub issue, opening a PR, or after one merges — and when asked which theme something belongs to, or why a Linear theme looks idle while work is shipping.
---

# Linking GitHub work to Linear themes

**Linear tracks themes; GitHub tracks work.** One Linear issue spans many
GitHub issues and many PRs. Mirroring them one-to-one makes both lists
worthless — Linear stops being a place to see shape and becomes a slower copy
of the GitHub tracker.

- Team prefix: `ALE`. Projects: **Constellations** (this repo, `stel`) and
  **Astrolabe** (the downstream data project).
- Themes are Linear issues in those projects, e.g. "AI harness feedback loop",
  "Retrieval quality", "Backend reach".

## Find the theme

Never work from a list of themes written down anywhere — it goes stale. Query:

```
list_issues(project="Constellations", state="Backlog|Todo|In Progress")
```

Match on what the work is *about*, not on which files it touches. If nothing
fits, say so and ask rather than inventing a theme; a new theme is a decision
about how the project is shaped, not a side effect of filing an issue.

## The three links, and which are reliable

| direction | mechanism | reliable? |
|---|---|---|
| GitHub issue → theme | `Theme: ALE-nn` + URL in the issue body | yes, you write it |
| PR → theme | bare `ALE-nn` in the PR body | best-effort; Linear may or may not surface it |
| theme → GitHub | a comment on the Linear issue naming the issue/PR | **yes — this is the one that always works** |

Branches here are named `feat/<issue>-<slug>` and carry no `ALE-` identifier,
so **Linear's automatic branch linking never fires.** That is the whole reason
the theme looks idle while work ships. Do not rely on the integration noticing;
post the comment.

## The trap: closing keywords

Write the identifier bare — `ALE-45`, or `Theme: ALE-45`.

**Never** `Fixes ALE-45`, `Closes ALE-45`, or `Resolves ALE-45`. Those are
Linear's completion keywords: merging one PR would mark the whole theme done
and bury every GitHub issue still open underneath it. A theme is finished when
a human decides it is, not when one of its PRs lands.

The same word is correct on the GitHub side — `Closes #380` is right when the
PR genuinely satisfies that issue. The keyword is safe for a GitHub issue and
wrong for a Linear theme, which is exactly why it is easy to get wrong.

## What to write

**Filing a GitHub issue** — first line of the body:

```
Theme: ALE-45 — AI harness feedback loop
https://linear.app/alex-noonan/issue/ALE-45/ai-harness-feedback-loop
```

Then comment on the theme:

```
save_comment(issue="ALE-45", body="GitHub #380 — candidate judgments and
promotion (#329 phase 3). https://github.com/.../issues/380")
```

**Opening a PR** — in the Tracking section the template already provides:

```
- Closes #380
- Theme: ALE-45
```

**After a PR merges** — one line on the theme saying what actually shipped:

```
save_comment(issue="ALE-45", body="Shipped in #387: candidate retrieval
judgments derived from the transcript corpus. #380 remains open for promotion
and classification labels.")
```

## Keep it proportionate

A theme wants a line per GitHub issue and per merged PR — enough to see what
moved and what remains. It does not want a running log of commits, review
rounds, or CI results. If a theme's comment history is longer than its
description, the traffic has stopped being signal.

## When the theme is genuinely done

Say so in a comment naming the evidence, and let a human close it. If several
GitHub issues under it are still open, it is not done — that is the situation
the closing-keyword rule exists to prevent.
