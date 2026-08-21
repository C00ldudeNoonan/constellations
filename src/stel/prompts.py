"""Named, versioned, immutable prompt artifacts (issue #303).

A prompt is a program input that changes the output, exactly like the SQL in a
transform — so it gets what code already gets: a name, a version, a diff in
review, and no editing once released. The analogy that fits is a database
migration: each version is a file, referenced explicitly, and improving one
means writing the next version rather than changing a released one.

```yaml
  llm:
    prompt: { name: signal_classify, version: v3 }   # prompts/signal_classify/v3.md
```

Inline `prompt: "..."` keeps working. It is right for quick projects and
examples, and this is an additional form rather than a replacement.

**Version resolution is explicit and required.** There is deliberately no
`latest` pointer: a moving reference reintroduces exactly the mutable-prompt
problem the versions exist to solve, since two runs of the same committed
project would resolve to different text.

**Resolved at compile time**, so a missing or misspelled version fails before
any source discovery, credential resolution, or provider call — a typo costs
nothing rather than costing a corpus.
"""

from __future__ import annotations

import re
import stat
from dataclasses import dataclass
from pathlib import Path

from .config.model import LLMTransformConfig, PromptRef
from .paths import is_within_project

# Where versioned prompts live, relative to the project directory.
PROMPTS_DIRNAME = "prompts"
PROMPT_SUFFIX = ".md"

# Name and version are path segments, so they are restricted to a conservative
# charset up front rather than sanitized later: a traversal or an absolute
# path must fail at config load, not resolve to a file outside the project.
_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class PromptError(Exception):
    """A prompt reference could not be resolved."""


@dataclass(frozen=True, slots=True)
class ResolvedPrompt:
    """Prompt text plus the identity that names it.

    `name`/`version` are None for an inline prompt: there is nothing stable to
    record, which is precisely the gap versioned prompts close.
    """

    text: str
    name: str | None = None
    version: str | None = None

    def identity(self) -> dict[str, str | None]:
        """Artifact-safe descriptor — never the text (issue #303, rule 5)."""
        return {"prompt_name": self.name, "prompt_version": self.version}


def validate_prompt_segment(value: str, *, label: str) -> str:
    if not _SEGMENT.match(value):
        raise ValueError(
            f"prompt {label} {value!r} is invalid: it becomes a path segment, "
            "so it must start with a letter or digit and contain only "
            "letters, digits, underscores, and hyphens"
        )
    return value


def verify_project_path(path: Path, project_dir: Path, *, what: str) -> Path:
    """Reject a path that leaves the project, by any route.

    Checking only the final component is not enough: a symlinked `prompts/` or
    `prompts/<name>/` leaves the `<version>.md` inside it an ordinary regular
    file, so the leaf check passes while the read lands outside the reviewed
    tree — and that text goes to an inference provider (Codex review, #334).
    Every component from the project root down is checked, and the resolved
    path must still be inside the project.
    """
    current = project_dir
    try:
        parts = path.relative_to(project_dir).parts
    except ValueError:
        parts = ()
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise PromptError(
                f"Refusing to use {what} at {path}: '{part}' is a symlink. "
                "stel confines project-configured paths to the project."
            )
    if not is_within_project(path.parent, project_dir):
        raise PromptError(
            f"Refusing to use {what} at {path}: it resolves outside the "
            "project directory."
        )
    return path


def prompt_path(ref: PromptRef, project_dir: Path) -> Path:
    """The file a reference names. Charset-validated at config load."""
    return (
        project_dir / PROMPTS_DIRNAME / ref.name / f"{ref.version}{PROMPT_SUFFIX}"
    )


def resolve_prompt(
    config: LLMTransformConfig, project_dir: Path, *, model_name: str
) -> ResolvedPrompt:
    """Resolve a model's prompt to text plus its identity.

    Raises `PromptError` for a missing version, an unreadable file, or an
    empty one — all of which are typos worth catching at compile time.
    """
    prompt = config.prompt
    if isinstance(prompt, str):
        return ResolvedPrompt(text=prompt)

    path = verify_project_path(
        prompt_path(prompt, project_dir),
        project_dir,
        what=f"prompt {prompt.name}/{prompt.version}",
    )
    if not path.exists() and not path.is_symlink():
        available = _available_versions(prompt.name, project_dir)
        hint = (
            f" Available versions of '{prompt.name}': {', '.join(available)}."
            if available
            else f" No versions of '{prompt.name}' exist yet."
        )
        raise PromptError(
            f"Model '{model_name}' references prompt "
            f"{prompt.name}/{prompt.version}, but {path} does not exist.{hint}"
        )
    # Same rule as project configuration: a regular, non-symlink file only, so
    # a prompt cannot be redirected outside the reviewed project tree.
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise PromptError(
            f"Refusing to read prompt {prompt.name}/{prompt.version} at {path}: "
            "expected a regular non-symlink file."
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise PromptError(
            f"Prompt {prompt.name}/{prompt.version} at {path} is empty; a "
            "prompt with no instruction is a typo, not a valid version."
        )
    return ResolvedPrompt(text=text, name=prompt.name, version=prompt.version)


def _available_versions(name: str, project_dir: Path) -> list[str]:
    directory = project_dir / PROMPTS_DIRNAME / name
    if not directory.is_dir():
        return []
    return sorted(
        item.stem
        for item in directory.iterdir()
        if item.is_file() and item.suffix == PROMPT_SUFFIX
    )
