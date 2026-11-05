"""A backend's identity covers the options definitions it reaches, no more (issue #572).

stel 0.24.0 changed the LLM `temperature` default in `backends/options.py`. The
JSON backend never reads it, but its identity hashed that whole module, so
`sec_document_registry` re-keyed on 20,289 of 20,289 rows and the change
cascaded through ten models, a paid embed step among them.
"""
from __future__ import annotations

import inspect
from collections.abc import Iterator

import pytest

import stel.backends.options as options_module
from stel.backends import base as base_module
from stel.backends.base import _dependency_source_digests
from stel.backends.options_identity import (
    CLOSURE_KEY,
    OPTIONS_MODULE,
    SHARED_VALIDATION_PATH,
    options_closure,
    options_imports,
)

PACKAGE = "stel"
MODULE_PACKAGE = "stel.backends"

SYNTHETIC = '''\
"""Options for several backends."""
from __future__ import annotations

from pydantic import BaseModel

from ..budget import LLMBudgetConfig
from ..hashing import HASH_DIGEST_SIZE

_STRICT = {"extra": "forbid"}
_LIMIT = 10


def _check(value):
    return value < _LIMIT


def _unrelated():
    return "unrelated"


class _Base(BaseModel):
    model_config = _STRICT


class Json(_Base):
    fields: list[str] | None = None


class Html(_Base):
    selector: str = "body"

    def validate_selector(self):
        return _check(len(self.selector))


class Llm(_Base):
    budget: LLMBudgetConfig | None = None
    temperature: float = 0.0
'''


def _closure(
    source: str,
    *,
    deep: set[str],
    shallow: frozenset[str] = frozenset(),
):
    return options_closure(
        source,
        module_package=MODULE_PACKAGE,
        package=PACKAGE,
        deep=frozenset(deep),
        shallow=shallow,
    )


def _edit(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, f"edit target {old!r} must occur exactly once"
    return source.replace(old, new)


def test_an_edit_to_another_backends_model_leaves_this_backends_digest_alone() -> None:
    """The incident: the LLM model changed and the JSON identity moved."""
    before = _closure(SYNTHETIC, deep={"Json"})
    after = _closure(
        _edit(SYNTHETIC, "temperature: float = 0.0", "temperature: float | None = None"),
        deep={"Json"},
    )
    assert after.digest == before.digest


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param(
            "fields: list[str] | None = None", "fields: list[str] = []", id="own"
        ),
        pytest.param(
            '_STRICT = {"extra": "forbid"}',
            '_STRICT = {"extra": "allow"}',
            id="named-constant",
        ),
        pytest.param(
            "model_config = _STRICT",
            "model_config = _STRICT\n    flag = 1",
            id="base-class",
        ),
        pytest.param("_LIMIT = 10", "_LIMIT = 11", id="constant-behind-a-helper"),
        pytest.param("return value < _LIMIT", "return value <= _LIMIT", id="helper"),
    ],
)
def test_an_edit_to_anything_a_model_names_moves_its_digest(old: str, new: str) -> None:
    """Every definition the model reaches by name must count, or a real
    behavior change would leave the old rows looking current."""
    deep = {"Json", "Html"}
    before = _closure(SYNTHETIC, deep=deep)
    after = _closure(_edit(SYNTHETIC, old, new), deep=deep)
    assert after.digest != before.digest


def test_an_edit_to_a_helper_moves_only_the_models_that_reach_it() -> None:
    edited = _edit(SYNTHETIC, "return value < _LIMIT", "return value <= _LIMIT")
    assert _closure(edited, deep={"Html"}).digest != _closure(SYNTHETIC, deep={"Html"}).digest
    assert _closure(edited, deep={"Json"}).digest == _closure(SYNTHETIC, deep={"Json"}).digest


def test_a_decorator_belongs_to_the_definition_it_decorates() -> None:
    # Only an argument changes: the decorator's name, which any definition that
    # names it would already notice, stays the same.
    source = "def _wrap(n):\n    return lambda f: f\n\n\n" + SYNTHETIC.replace(
        "def _check", "@_wrap(1)\ndef _check"
    )
    edited = source.replace("@_wrap(1)", "@_wrap(2)")
    assert _closure(edited, deep={"Html"}).digest != _closure(source, deep={"Html"}).digest


def test_a_module_level_statement_moves_every_digest() -> None:
    """A statement that is not a definition runs whichever model is asked for."""
    with_effect = SYNTHETIC + "\nREGISTRY = {}\nREGISTRY['json'] = Json\n"
    # The same names on both sides, so only the statement's text can show it.
    edited = with_effect.replace("REGISTRY['json']", "REGISTRY['xml']")
    assert _closure(edited, deep={"Json"}).digest != _closure(with_effect, deep={"Json"}).digest


def test_a_shallow_root_hashes_its_own_text_without_following_its_references() -> None:
    """`validate_backend_options` names the LLM model; following that would hand
    every backend the LLM model back."""
    source = SYNTHETIC + "\n\ndef validate(name):\n    return isinstance(name, Llm)\n"
    shallow = frozenset({"validate"})
    before = _closure(source, deep={"Json"}, shallow=shallow)

    llm_edit = _edit(source, "temperature: float = 0.0", "temperature: float = 0.5")
    assert _closure(llm_edit, deep={"Json"}, shallow=shallow).digest == before.digest

    own_edit = source.replace("isinstance(name, Llm)", "isinstance(name, Html)")
    assert _closure(own_edit, deep={"Json"}, shallow=shallow).digest != before.digest


def test_modules_the_followed_definitions_import_are_reported_for_hashing() -> None:
    assert _closure(SYNTHETIC, deep={"Llm"}).modules == {"stel.budget"}
    assert _closure(SYNTHETIC, deep={"Json"}).modules == frozenset()


def test_a_name_the_module_does_not_define_still_perturbs_the_digest() -> None:
    assert (
        _closure(SYNTHETIC, deep={"Json", "Gone"}).digest
        != _closure(SYNTHETIC, deep={"Json"}).digest
    )


@pytest.mark.parametrize(
    ("statement", "names", "whole"),
    [
        ("from .options import A, B", {"A", "B"}, False),
        ("from stel.backends.options import A", {"A"}, False),
        ("from . import options", set(), True),
        ("from .options import *", set(), True),
        ("import stel.backends.options", set(), True),
        ("from .base import A", set(), False),
    ],
)
def test_what_a_module_takes_from_the_options_module(
    statement: str, names: set[str], whole: bool
) -> None:
    taken = options_imports(statement, module_package=MODULE_PACKAGE)
    assert taken.names == names
    assert taken.whole is whole


# --- the shipped modules -------------------------------------------------


def _real_options_source() -> str:
    return inspect.getsource(options_module)


def _real_closure(source: str, root: str):
    return _closure(source, deep={root}, shallow=SHARED_VALIDATION_PATH)


def test_the_0_24_0_llm_edit_would_not_have_re_keyed_the_other_backends() -> None:
    """Issue #572, on the real module: the `temperature` default and the
    validator skip that shipped in 0.24.0 move the LLM digest and no other."""
    source = _real_options_source()
    edited = _edit(
        source,
        "temperature: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)",
        "temperature: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)",
    )
    roots = {
        "json": "JsonBackendOptions",
        "markdown": "MarkdownBackendOptions",
        "pdf": "PdfBackendOptions",
        "html": "HtmlBackendOptions",
        "email": "EmailBackendOptions",
        "llm": "LLMBackendOptions",
    }
    moved = {
        backend
        for backend, root in roots.items()
        if _real_closure(edited, root).digest != _real_closure(source, root).digest
    }
    assert moved == {"llm"}


def test_the_shared_validation_path_still_re_keys_every_backend() -> None:
    """Shared code is shared: changing how any options are validated is a
    change to all of them, whichever model it was written for."""
    source = _real_options_source()
    edited = _edit(source, "exclude_unset=True,", "exclude_unset=False,")
    for root in ("JsonBackendOptions", "LLMBackendOptions"):
        assert _real_closure(edited, root).digest != _real_closure(source, root).digest


@pytest.fixture
def fresh_dependency_digests() -> Iterator[None]:
    """The digests are cached per process; this module edits the source they read."""
    _dependency_source_digests.cache_clear()
    yield
    _dependency_source_digests.cache_clear()


def _digests(module: str, roots: tuple[str, ...]) -> dict[str, str | None]:
    return dict(_dependency_source_digests(module, option_roots=roots))


@pytest.mark.usefixtures("fresh_dependency_digests")
def test_the_json_backend_no_longer_depends_on_modules_only_the_llm_options_import() -> None:
    json_digests = _digests("stel.backends.json_backend", ("JsonBackendOptions",))
    llm_digests = _digests("stel.backends.llm_backend", ("LLMBackendOptions",))

    assert OPTIONS_MODULE not in json_digests
    assert CLOSURE_KEY in json_digests
    # `stel.budget` is not here: `base.py` imports it for every backend.
    for module in ("stel.credentials", "stel.endpoints"):
        assert module not in json_digests
        assert module in llm_digests
    assert all(digest is not None for digest in json_digests.values())


@pytest.mark.usefixtures("fresh_dependency_digests")
def test_a_backend_identity_follows_an_edit_to_its_own_options_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through the dependency walk itself, not only the closure function."""
    original = base_module._source_text

    def edited_options(obj: object) -> str | None:
        text = original(obj)
        if text is not None and getattr(obj, "__name__", "") == OPTIONS_MODULE:
            return text.replace("default=2048", "default=1024")
        return text

    before_json = _digests("stel.backends.json_backend", ("JsonBackendOptions",))
    before_llm = _digests("stel.backends.llm_backend", ("LLMBackendOptions",))
    _dependency_source_digests.cache_clear()
    monkeypatch.setattr(base_module, "_source_text", edited_options)

    assert _digests("stel.backends.json_backend", ("JsonBackendOptions",)) == before_json
    after_llm = _digests("stel.backends.llm_backend", ("LLMBackendOptions",))
    assert after_llm[CLOSURE_KEY] != before_llm[CLOSURE_KEY]
