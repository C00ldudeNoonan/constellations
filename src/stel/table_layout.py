"""Column sizing for the fixed-width tables the CLI prints.

A literal width such as ``{name:<22}`` has no separator: a name that reaches
the width runs into the next column, and a longer one shifts the rest of its
row out of line with the header (issue #819). Truncating is not an answer,
because a truncated name is ambiguous in a captured log. So a column is sized
from the values it will hold, and never narrower than the width the table has
always used, which keeps the output for ordinary names byte-for-byte unchanged.
"""

from collections.abc import Iterable

# The model-name column of the live ledger and of every model table. The
# ledger prints one line at a time, so its width comes from the selected names
# up front rather than from rows already printed.
LEDGER_NAME_MINIMUM = 24

# `stel orphans`, `stel pending` and the `watch` re-run lines (issue #822). Each
# is its historical name width including any space that used to follow it, so
# the separating space now comes from `column_width` rather than a literal.
ORPHAN_NAME_MINIMUM = 32
PENDING_NAME_MINIMUM = 33
RERUN_NAME_MINIMUM = 23


def column_width(values: Iterable[str], *, minimum: int) -> int:
    """Width of a left-aligned column: the longest value plus one separating
    space, and at least ``minimum`` (the table's historical width)."""
    longest = max((len(value) for value in values), default=0)
    return max(minimum, longest + 1)
