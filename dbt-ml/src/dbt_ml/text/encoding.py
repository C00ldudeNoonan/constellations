from __future__ import annotations

from typing import cast

from ..optional_dependencies import import_optional_dependency


def clean_encoding(text: str) -> str:
    """Fix mojibake and common encoding artifacts using ftfy.

    Turns things like 'a-tilde-euro-tm' back into a real apostrophe, repairs
    Latin-1-in-UTF-8 confusion,
    normalizes line endings, and strips control characters. A one-liner that
    saves hours of grep-driven debugging on real-world data.
    """
    if not text:
        return text
    ftfy = import_optional_dependency(
        "ftfy", extra="text", feature="Text encoding cleanup"
    )

    return cast(str, ftfy.fix_text(text))
