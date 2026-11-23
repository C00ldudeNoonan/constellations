"""The constants `cli.py` duplicates as literals (issue #818, ALE-790).

A Click option declaration runs when `stel.cli` is imported, so importing its
choices or default from the module that owns them would load the whole warehouse
and retrieval stack on every `stel --help`. `cli.py` carries literals instead,
and this file pins each to its source of truth so a rename or a new member fails
a test rather than drifting. (The grants literals are pinned in
`test_grants_cli.py`.)
"""

from __future__ import annotations

from stel import cli
from stel.graph_export import GRAPH_FORMATS
from stel.retrieval.servability import (
    DEFAULT_CONTEXT_TIMEOUT_SECONDS,
    MAX_CONTEXT_TIMEOUT_SECONDS,
)
from stel.search import SearchFilterOperator, SearchMode
from stel.suggest import DEFAULT_MIN_EVIDENCE


def test_search_literals_match_the_enums() -> None:
    assert cli._SEARCH_MODES == [mode.value for mode in SearchMode]
    assert cli._DEFAULT_SEARCH_MODE == SearchMode.HYBRID.value
    assert cli._SEARCH_FILTER_OPERATORS == [op.value for op in SearchFilterOperator]
    assert cli._JSON_ARRAY_FILTER_OPERATORS == {
        SearchFilterOperator.IN.value,
        SearchFilterOperator.ARRAY_CONTAINS_ANY.value,
    }


def test_graph_and_suggest_literals_match_their_modules() -> None:
    assert cli._GRAPH_FORMATS == GRAPH_FORMATS
    assert cli._DEFAULT_MIN_EVIDENCE == DEFAULT_MIN_EVIDENCE


def test_mcp_timeout_literals_match_the_servability_bounds() -> None:
    assert cli._MAX_CONTEXT_TIMEOUT_SECONDS == MAX_CONTEXT_TIMEOUT_SECONDS
    assert cli._DEFAULT_CONTEXT_TIMEOUT_SECONDS == DEFAULT_CONTEXT_TIMEOUT_SECONDS
