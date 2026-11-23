"""NLP provider registry, provider contract, and language routing (issue #222).

Everything here runs on deterministic fakes with no network, no spaCy and no
language detector: the detector is injected where a test is about routing and
the real `langdetect` is exercised once, behind the `text` extra.
"""
from __future__ import annotations

import sys
import textwrap
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from stel.adapters import parse_warehouse_config
from stel.text import nlp as nlp_core
from stel.text import nlp_registry, nlp_routing
from stel.text.nlp import (
    NLPDocument,
    NLPEntity,
    NLPEntityOptions,
    NLPError,
    NLPIdentity,
    NLPProvider,
    NLPToken,
    NLPTokenOptions,
)
from stel.text.nlp_registry import (
    NLP_PROVIDER_CONTRACT_VERSION,
    NLPModelSpec,
    NLPProviderRegistration,
    NLPProviderRegistryError,
    discover_nlp_providers,
    entry_point_group,
    get_nlp_registration,
    list_nlp_providers,
    register_nlp_provider,
)
from stel.text.nlp_routing import (
    NLPLanguageRouting,
    NLPRoutingError,
    decide_route,
    observed_language,
)
from stel.text.transforms import nlp_entities, nlp_tokens
from stel.transforms import TransformContext

_ALL = frozenset(("tokens", "lemmas", "pos", "entities", "sentences"))


class _FakeProvider:
    """Marks every token with its provider and model so a test can see which
    route analyzed a document."""

    def __init__(self, name: str, spec: NLPModelSpec) -> None:
        self.identity = NLPIdentity(
            provider=name,
            provider_version="1.0",
            model=spec.model,
            model_version="2026.1",
            language=spec.language,
        )
        self.piped: list[list[str]] = []

    def pipe(
        self, texts: Iterable[str], *, batch_size: int
    ) -> Iterable[NLPDocument]:
        batch = list(texts)
        self.piped.append(batch)
        for text in batch:
            token = NLPToken(
                index=0,
                sentence_index=0,
                start=0,
                end=len(text),
                text=text,
                lemma=f"{self.identity.model}:{text}",
                pos="X",
                tag="X",
                is_stop=False,
                is_alpha=True,
                is_space=False,
            )
            entity = NLPEntity(
                index=0, sentence_index=0, start=0, end=len(text), label="ORG", text=text
            )
            yield NLPDocument(tokens=(token,), entities=(entity,))


class _Loads:
    """Records every load a registered fake provider receives."""

    def __init__(self) -> None:
        self.specs: list[NLPModelSpec] = []
        self.providers: list[_FakeProvider] = []

    def loader(self, name: str) -> Callable[[NLPModelSpec], NLPProvider]:
        def load(spec: NLPModelSpec) -> NLPProvider:
            self.specs.append(spec)
            provider = _FakeProvider(name, spec)
            self.providers.append(provider)
            return provider

        return load


def _register(
    name: str,
    loads: _Loads | None = None,
    *,
    capabilities: frozenset[str] = _ALL,
    languages: frozenset[str] | None = None,
) -> _Loads:
    loads = loads or _Loads()
    register_nlp_provider(
        NLPProviderRegistration(
            name=name,
            loader=loads.loader(name),
            capabilities=capabilities,
            languages=languages,
        )
    )
    return loads


def _ctx(options: dict[str, Any]) -> TransformContext:
    return TransformContext(
        project_dir=Path("."),
        profile_name="test",
        target_name="dev",
        warehouse=parse_warehouse_config(
            {"type": "duckdb", "path": "./test.duckdb", "schema": "main"}
        ),
        llm=None,
        options=options,
    )


def _routing_options(**routing: Any) -> dict[str, Any]:
    return {"language_routing": routing}


_TWO_ROUTES: dict[str, Any] = {
    "en": {"provider": "fake", "model": "en-model"},
    "de": {"provider": "fake", "model": "de-model"},
}


@pytest.fixture
def detector(monkeypatch: pytest.MonkeyPatch) -> dict[str, str | None]:
    """Detect by lookup so routing tests do not depend on `langdetect`."""
    table: dict[str, str | None] = {}
    monkeypatch.setattr(nlp_routing, "detect_language", lambda text: table.get(text))
    return table


# --- provider contract -------------------------------------------------------


def test_spacy_is_a_registered_provider_with_declared_capabilities() -> None:
    registration = get_nlp_registration("spacy")

    assert registration.capabilities == _ALL
    assert "confidence" not in registration.capabilities
    # The installed model decides the language, so it is checked at load time.
    assert registration.languages is None
    assert "spacy" in list_nlp_providers()


@pytest.mark.parametrize("provider_name", ["spacy", "fake"])
def test_every_registered_provider_honours_the_contract(
    monkeypatch: pytest.MonkeyPatch, provider_name: str
) -> None:
    """The conformance suite a provider must pass, run over the built-in and a
    fake, with deterministic doubles and no network."""
    if provider_name == "fake":
        _register("fake")
    else:
        # `_load_spacy` looks `_spacy_provider` up at call time.
        monkeypatch.setattr(nlp_core, "_spacy_provider", _spacy_double)
    registration = get_nlp_registration(provider_name)
    route = nlp_routing.NLPRoute(provider=provider_name, model="m", language="en")

    provider = nlp_core.load_nlp_route(route, "en")
    documents = list(provider.pipe(["a", "bb", "ccc"], batch_size=2))

    assert registration.name == provider_name
    assert provider.identity.provider == provider_name
    assert provider.identity.language == "en"
    # one analyzed document per input text, in order
    assert len(documents) == 3
    assert documents[2].tokens[0].end == 3


def _spacy_double(model: str, language: str, disable: tuple[str, ...]) -> NLPProvider:
    return _FakeProvider(
        "spacy", NLPModelSpec(model=model, language=language, disable=disable)
    )


def test_a_loader_that_returns_another_providers_identity_is_rejected() -> None:
    register_nlp_provider(
        NLPProviderRegistration(
            name="liar",
            loader=lambda spec: _FakeProvider("someone-else", spec),
            capabilities=_ALL,
            languages=None,
        )
    )

    with pytest.raises(NLPError, match="identifies as 'someone-else'"):
        nlp_core.load_nlp_route(nlp_routing.NLPRoute(provider="liar", model="m"), "en")


@pytest.mark.parametrize(
    ("registration", "message"),
    [
        (dict(name="Bad Name"), "name must match"),
        (dict(name="p", capabilities=frozenset()), "at least one capability"),
        (
            dict(name="p", capabilities=frozenset(("tokens", "telepathy"))),
            "unknown capabilities",
        ),
        (dict(name="p", languages=frozenset()), "non-empty set of normalized codes"),
        (dict(name="p", languages=frozenset(("EN",))), "normalized codes"),
    ],
)
def test_registration_is_validated_when_registered(
    registration: dict[str, Any], message: str
) -> None:
    fields: dict[str, Any] = {
        "loader": lambda spec: _FakeProvider("p", spec),
        "capabilities": _ALL,
        "languages": None,
        **registration,
    }

    with pytest.raises(NLPProviderRegistryError, match=message):
        register_nlp_provider(NLPProviderRegistration(**fields))


def test_a_name_cannot_be_registered_twice() -> None:
    _register("dup")

    with pytest.raises(NLPProviderRegistryError, match="already registered"):
        _register("dup")


def test_registering_the_identical_registration_again_is_idempotent() -> None:
    first = get_nlp_registration("spacy")

    register_nlp_provider(first)

    assert get_nlp_registration("spacy") is first


def test_unknown_provider_lists_what_is_available() -> None:
    with pytest.raises(NLPProviderRegistryError, match=r"'nope' is not registered.*spacy"):
        get_nlp_registration("nope")


# --- entry-point discovery ---------------------------------------------------

_PLUGIN_SOURCE = """
from stel.text import NLPProviderRegistration

def _load(spec):
    raise RuntimeError("never loaded in these tests")

registration = NLPProviderRegistration(
    name="{name}",
    loader=_load,
    capabilities=frozenset(("tokens", "lemmas", "pos")),
    languages=frozenset(("de", "fr")),
)
"""


def _install_plugin(
    site_dir: Path,
    *,
    distribution: str,
    module: str,
    name: str,
    group: str,
    entry_name: str | None = None,
    attribute: str = "registration",
    source: str | None = None,
) -> None:
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / f"{module}.py").write_text(
        textwrap.dedent(source or _PLUGIN_SOURCE.format(name=name)), encoding="utf-8"
    )
    dist_info = site_dir / f"{distribution}-0.1.0.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 0.1.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        f"[{group}]\n{entry_name or name} = {module}:{attribute}\n", encoding="utf-8"
    )


@pytest.fixture
def plugin_modules() -> Iterable[None]:
    before = set(sys.modules)
    yield
    for module_name in set(sys.modules) - before:
        if module_name.startswith("nlpplugin_"):
            del sys.modules[module_name]


def _discover(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(site))
    discover_nlp_providers(force=True)


def test_entry_point_group_carries_the_contract_version() -> None:
    assert entry_point_group() == f"stel.nlp_providers.v{NLP_PROVIDER_CONTRACT_VERSION}"


def test_a_separately_packaged_provider_registers_without_core_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plugin_modules: None
) -> None:
    """The issue's first done-when: a third party registers through an entry
    point, and then routes and capability checks apply to it unchanged."""
    _install_plugin(
        tmp_path / "site",
        distribution="nlp-plugin-good",
        module="nlpplugin_good",
        name="acme",
        group=entry_point_group(),
    )
    _discover(tmp_path / "site", monkeypatch)

    registration = get_nlp_registration("acme")
    assert registration.languages == frozenset(("de", "fr"))
    assert nlp_registry._PLUGIN_DISTRIBUTIONS["acme"] == "nlp-plugin-good"
    # no `entities` capability: the entity transform rejects it at compile time
    with pytest.raises(ValidationError, match="does not provide \\['entities'\\]"):
        NLPEntityOptions.model_validate({"provider": "acme", "model": "m", "language": "de"})
    NLPTokenOptions.model_validate({"provider": "acme", "model": "m", "language": "de"})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(entry_name="othername"), "must load an NLPProviderRegistration"),
        (dict(name="spacy"), "claimed by both 'built-in' and 'nlp-plugin-bad'"),
        (
            dict(source="raise RuntimeError('secret-path /home/x')\n"),
            r"failed to load \[RuntimeError\]",
        ),
        (dict(source="registration = 1\n"), "must load an NLPProviderRegistration"),
    ],
)
def test_broken_plugins_fail_closed_with_the_distribution_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: None,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    params: dict[str, Any] = {"name": "acme", **kwargs}
    _install_plugin(
        tmp_path / "site",
        distribution="nlp-plugin-bad",
        module="nlpplugin_bad",
        group=entry_point_group(),
        **params,
    )

    with pytest.raises(NLPProviderRegistryError, match=message) as caught:
        _discover(tmp_path / "site", monkeypatch)

    assert "secret-path" not in str(caught.value)
    assert "nlp-plugin-bad" in str(caught.value)


def test_two_distributions_cannot_claim_one_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plugin_modules: None
) -> None:
    for distribution, module in (("nlp-a", "nlpplugin_a"), ("nlp-b", "nlpplugin_b")):
        _install_plugin(
            tmp_path / "site",
            distribution=distribution,
            module=module,
            name="acme",
            group=entry_point_group(),
        )

    with pytest.raises(NLPProviderRegistryError, match="claimed by both 'nlp-a' and 'nlp-b'"):
        _discover(tmp_path / "site", monkeypatch)


def test_a_plugin_for_another_contract_version_is_reported_as_such(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plugin_modules: None
) -> None:
    _install_plugin(
        tmp_path / "site",
        distribution="nlp-plugin-old",
        module="nlpplugin_old",
        name="oldvendor",
        group="stel.nlp_providers.v0",
    )
    _discover(tmp_path / "site", monkeypatch)

    with pytest.raises(NLPProviderRegistryError, match="targets contract v0"):
        get_nlp_registration("oldvendor")
    assert "nlpplugin_old" not in sys.modules  # an incompatible plugin is never imported


# --- preflight (compile-time) validation -------------------------------------


def test_unknown_provider_fails_validation_before_anything_loads() -> None:
    with pytest.raises(ValidationError, match="'nope' is not registered"):
        NLPTokenOptions.model_validate({"provider": "nope"})


def test_missing_capability_fails_validation() -> None:
    _register("tokens-only", capabilities=frozenset(("tokens", "lemmas", "pos")))

    with pytest.raises(ValidationError, match="does not provide \\['entities'\\]"):
        NLPEntityOptions.model_validate({"provider": "tokens-only"})
    NLPTokenOptions.model_validate({"provider": "tokens-only"})


def test_unsupported_language_fails_validation() -> None:
    _register("german-only", languages=frozenset(("de",)))

    with pytest.raises(ValidationError, match="does not support language 'en'"):
        NLPTokenOptions.model_validate({"provider": "german-only", "language": "en"})
    NLPTokenOptions.model_validate({"provider": "german-only", "language": "de"})


def test_each_route_is_checked_and_named_in_the_error() -> None:
    _register("german-only", languages=frozenset(("de",)))
    _register("fake")

    with pytest.raises(ValidationError, match="route 'fr': provider 'german-only'"):
        NLPTokenOptions.model_validate(
            _routing_options(
                routes={
                    "de": {"provider": "german-only", "model": "m"},
                    "fr": {"provider": "german-only", "model": "m"},
                }
            )
        )


def test_fallback_route_is_checked_too() -> None:
    _register("german-only", languages=frozenset(("de",)))
    _register("fake")

    with pytest.raises(ValidationError, match="fallback route: provider 'german-only'"):
        NLPTokenOptions.model_validate(
            _routing_options(
                routes={"en": {"provider": "fake", "model": "m"}},
                on_unrouted="fallback",
                fallback={"provider": "german-only", "model": "m", "language": "en"},
            )
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"model": "en_core_web_sm"}, "move \\['model'\\] into routes"),
        ({"provider": "spacy", "disable": ["ner"]}, "move \\['disable', 'provider'\\]"),
    ],
)
def test_routing_replaces_the_single_model(
    options: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        NLPTokenOptions.model_validate(
            {**options, **_routing_options(routes={"en": {"model": "m"}})}
        )


@pytest.mark.parametrize(
    ("routing", "message"),
    [
        ({"routes": {}}, "at least 1 item"),
        ({"routes": {"english": {"model": "m"}}}, "must be a language code"),
        ({"routes": {"en": {"model": "m"}, "EN": {"model": "m"}}}, "declared twice"),
        (
            {"routes": {"en": {"model": "m"}}, "on_unrouted": "fallback"},
            "requires a fallback route",
        ),
        (
            {
                "routes": {"en": {"model": "m"}},
                "fallback": {"model": "m", "language": "xx"},
            },
            "neither on_undetected nor on_unrouted uses it",
        ),
        (
            {
                "routes": {"en": {"model": "m"}},
                "on_undetected": "fallback",
                "fallback": {"model": "m"},
            },
            "fallback.language is required",
        ),
        ({"routes": {"en": {"model": "m"}}, "on_unrouted": "ignore"}, "fail"),
        ({"routes": {"en": {"model": "m"}}, "surprise": 1}, "Extra inputs"),
        ({"routes": {"en": {"model": "m", "language": "english"}}}, "language code"),
    ],
)
def test_routing_config_is_strict(routing: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        NLPTokenOptions.model_validate(_routing_options(**routing))


def test_validation_does_not_load_any_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = _register("fake")

    NLPTokenOptions.model_validate(_routing_options(routes=_TWO_ROUTES))
    nlp_tokens.validate_options(_routing_options(routes=_TWO_ROUTES))

    assert loads.specs == []


# --- route decisions ----------------------------------------------------------


def _routing(**overrides: Any) -> NLPLanguageRouting:
    if "fake" not in nlp_registry._REGISTRY:
        _register("fake")
    return NLPLanguageRouting.model_validate({"routes": _TWO_ROUTES, **overrides})


def test_decision_prefers_the_exact_code_then_the_primary_subtag() -> None:
    routing = NLPLanguageRouting.model_validate(
        {"routes": {"zh-cn": {"model": "a"}, "zh": {"model": "b"}, "en": {"model": "c"}}}
    )

    assert decide_route(routing, document_id="d", observed="zh-cn").route_key == "zh-cn"
    assert decide_route(routing, document_id="d", observed="zh-tw").route_key == "zh"
    assert decide_route(routing, document_id="d", observed="en-gb").route_key == "en"


def test_a_route_for_a_subtag_language_expects_the_primary_language_model() -> None:
    routing = NLPLanguageRouting.model_validate({"routes": {"zh-cn": {"model": "a"}}})

    assert routing.route_language("zh-cn") == "zh"


@pytest.mark.parametrize(
    ("observed", "policy_field"),
    [(None, "on_undetected"), ("fr", "on_unrouted")],
)
@pytest.mark.parametrize(
    ("policy", "expected"), [("skip", None), ("fallback", "fallback")]
)
def test_each_policy_applies_to_its_own_case(
    observed: str | None, policy_field: str, policy: str, expected: str | None
) -> None:
    fallback = {"provider": "fake", "model": "multi", "language": "xx"}
    routing = _routing(
        **{policy_field: policy},
        **({"fallback": fallback} if policy == "fallback" else {}),
    )

    decision = decide_route(routing, document_id="d", observed=observed)

    assert decision.route_key == expected
    assert decision.observed_language == observed


def test_the_other_policy_is_not_consulted() -> None:
    routing = _routing(on_unrouted="skip")

    with pytest.raises(NLPRoutingError, match="has no determinable language"):
        decide_route(routing, document_id="doc-1", observed=None)


def test_fail_policy_names_the_document_and_language_not_its_text() -> None:
    routing = _routing()

    with pytest.raises(NLPRoutingError) as caught:
        decide_route(routing, document_id="doc-7", observed="fr")

    message = str(caught.value)
    assert "doc-7" in message
    assert "'fr'" in message
    assert "['de', 'en']" in message


def test_declared_language_is_normalized_and_blank_means_undetected() -> None:
    routing = _routing(language_field="lang")

    assert observed_language(routing, declared="EN_gb", text="x") == "en-gb"
    assert observed_language(routing, declared="  ", text="x") is None
    assert observed_language(routing, declared=None, text="x") is None


def test_detection_is_used_only_without_a_language_field() -> None:
    def forbidden(_text: str) -> str:
        raise AssertionError("must not detect when the language is declared")

    declared = _routing(language_field="lang")
    detected = _routing()

    assert observed_language(declared, declared="de", text="t", detect=forbidden) == "de"
    assert observed_language(detected, declared="de", text="t", detect=lambda _t: "FR") == "fr"
    assert observed_language(detected, declared=None, text="t", detect=lambda _t: None) is None


def test_the_real_detector_is_deterministic_and_routes() -> None:
    pytest.importorskip("langdetect")
    routing = _routing()
    english = "The Federal Reserve raised interest rates again at its latest meeting."
    german = "Die Bundesbank hat die Zinsen auf ihrer letzten Sitzung erneut angehoben."

    first = [observed_language(routing, declared=None, text=t) for t in (english, german)]
    second = [observed_language(routing, declared=None, text=t) for t in (english, german)]

    assert first == second == ["en", "de"]


# --- routed execution ----------------------------------------------------------


def _frame(rows: list[tuple[str, str, str | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "document_id": [row[0] for row in rows],
            "text": [row[1] for row in rows],
            "lang": [row[2] for row in rows],
            "year": list(range(len(rows))),
        }
    )


def test_mixed_language_documents_route_predictably_with_identity_recorded(
    detector: dict[str, str | None],
) -> None:
    loads = _register("fake")
    detector.update({"hello": "en", "hallo": "de", "world": "en"})
    upstream = _frame([("a", "hello", None), ("b", "hallo", None), ("c", "world", None)])
    options = {**_routing_options(routes=_TWO_ROUTES), "include_fields": ["year"]}

    tokens = nlp_tokens.run({"raw": upstream}, _ctx(options))

    # input order is preserved even though the routes ran as two batches
    assert tokens["document_id"].to_list() == ["a", "b", "c"]
    assert tokens["lemma"].to_list() == ["en-model:hello", "de-model:hallo", "en-model:world"]
    assert tokens["nlp_route"].to_list() == ["en", "de", "en"]
    assert tokens["nlp_input_language"].to_list() == ["en", "de", "en"]
    assert tokens["nlp_language_source"].to_list() == ["detected"] * 3
    assert tokens["nlp_model"].to_list() == ["en-model", "de-model", "en-model"]
    assert tokens["nlp_language"].to_list() == ["en", "de", "en"]
    assert tokens["nlp_provider"].to_list() == ["fake"] * 3
    # one load and one pipe per route, not per document
    assert sorted(spec.model for spec in loads.specs) == ["de-model", "en-model"]
    assert sorted(len(p.piped) for p in loads.providers) == [1, 1]
    assert [p.piped for p in loads.providers if p.identity.model == "en-model"] == [
        [["hello", "world"]]
    ]


def test_routing_is_reproducible(detector: dict[str, str | None]) -> None:
    _register("fake")
    detector.update({"hello": "en", "hallo": "de"})
    upstream = _frame([("a", "hello", None), ("b", "hallo", None)])
    options = _routing_options(routes=_TWO_ROUTES)

    first = nlp_tokens.run({"raw": upstream}, _ctx(options))
    second = nlp_tokens.run({"raw": upstream}, _ctx(options))

    assert first.equals(second)


def test_entities_are_routed_the_same_way(detector: dict[str, str | None]) -> None:
    _register("fake")
    detector.update({"hello": "en", "hallo": "de"})
    upstream = _frame([("a", "hello", None), ("b", "hallo", None)])

    entities = nlp_entities.run(
        {"raw": upstream},
        _ctx({**_routing_options(routes=_TWO_ROUTES), "include_text": True}),
    )

    assert entities["entity_text"].to_list() == ["hello", "hallo"]
    assert entities["nlp_route"].to_list() == ["en", "de"]
    assert entities["nlp_model"].to_list() == ["en-model", "de-model"]


def test_declared_language_column_routes_without_detection(
    detector: dict[str, str | None],
) -> None:
    _register("fake")
    detector.clear()  # a detector call would return None and fail the run
    upstream = _frame([("a", "hello", "EN"), ("b", "hallo", "de")])

    tokens = nlp_tokens.run(
        {"raw": upstream},
        _ctx(_routing_options(routes=_TWO_ROUTES, language_field="lang")),
    )

    assert tokens["nlp_route"].to_list() == ["en", "de"]
    assert tokens["nlp_language_source"].to_list() == ["declared", "declared"]


def test_skip_drops_the_document_and_the_rest_still_run(
    detector: dict[str, str | None],
) -> None:
    loads = _register("fake")
    detector.update({"hello": "en", "bonjour": "fr"})
    upstream = _frame([("a", "hello", None), ("b", "bonjour", None), ("c", "?", None)])

    tokens = nlp_tokens.run(
        {"raw": upstream},
        _ctx(
            _routing_options(
                routes=_TWO_ROUTES, on_unrouted="skip", on_undetected="skip"
            )
        ),
    )

    assert tokens["document_id"].to_list() == ["a"]
    assert [spec.model for spec in loads.specs] == ["en-model"]


def test_fallback_route_is_used_and_labelled(detector: dict[str, str | None]) -> None:
    loads = _register("fake")
    detector.update({"hello": "en", "bonjour": "fr"})
    upstream = _frame([("a", "hello", None), ("b", "bonjour", None), ("c", "?", None)])

    tokens = nlp_tokens.run(
        {"raw": upstream},
        _ctx(
            _routing_options(
                routes=_TWO_ROUTES,
                on_unrouted="fallback",
                on_undetected="fallback",
                fallback={"provider": "fake", "model": "multi", "language": "xx"},
            )
        ),
    )

    assert tokens["nlp_route"].to_list() == ["en", "fallback", "fallback"]
    assert tokens["nlp_input_language"].to_list() == ["en", "fr", None]
    assert tokens["nlp_model"].to_list() == ["en-model", "multi", "multi"]
    assert tokens["nlp_language"].to_list() == ["en", "xx", "xx"]
    fallback_load = next(spec for spec in loads.specs if spec.model == "multi")
    assert fallback_load.language == "xx"


def test_a_failing_document_stops_the_run_before_any_provider_loads(
    detector: dict[str, str | None],
) -> None:
    loads = _register("fake")
    detector.update({"hello": "en", "bonjour": "fr"})
    upstream = _frame([("a", "hello", None), ("b", "bonjour", None)])

    with pytest.raises(NLPRoutingError, match="Document 'b' has language 'fr'"):
        nlp_tokens.run({"raw": upstream}, _ctx(_routing_options(routes=_TWO_ROUTES)))

    assert loads.specs == []


def test_declared_language_column_must_exist_and_hold_strings() -> None:
    _register("fake")
    options = _ctx(_routing_options(routes=_TWO_ROUTES, language_field="lang"))

    with pytest.raises(ValueError, match="Expected language column 'lang'"):
        nlp_tokens.run(
            {"raw": pl.DataFrame({"document_id": ["a"], "text": ["x"]})}, options
        )
    with pytest.raises(ValueError, match="Language column 'lang' must contain strings"):
        nlp_tokens.run(
            {"raw": pl.DataFrame({"document_id": ["a"], "text": ["x"], "lang": [1]})},
            options,
        )


def test_routing_columns_are_reserved_and_absent_without_routing(
    detector: dict[str, str | None],
) -> None:
    _register("fake")
    detector.update({"hello": "en"})
    upstream = pl.DataFrame(
        {"document_id": ["a"], "text": ["hello"], "nlp_route": ["unsafe"]}
    )

    with pytest.raises(ValueError, match="collides with NLP output columns"):
        nlp_tokens.run(
            {"raw": upstream},
            _ctx({**_routing_options(routes=_TWO_ROUTES), "include_fields": ["nlp_route"]}),
        )

    # the same include is fine for an unrouted transform: its schema is unchanged
    _register("fake2")
    plain = nlp_tokens.run(
        {"raw": upstream},
        _ctx({"provider": "fake2", "include_fields": ["nlp_route"]}),
    )
    assert "nlp_input_language" not in plain.columns
    assert plain["nlp_route"].to_list() == ["unsafe"]


def test_empty_routed_input_returns_the_routed_schema_without_loading() -> None:
    loads = _register("fake")
    empty = pl.DataFrame(schema={"document_id": pl.String(), "text": pl.String()})

    output = nlp_tokens.run({"raw": empty}, _ctx(_routing_options(routes=_TWO_ROUTES)))

    assert output.is_empty()
    assert output.schema["nlp_route"] == pl.String()
    assert loads.specs == []


def test_routed_provider_must_return_one_document_per_text(
    detector: dict[str, str | None],
) -> None:
    class _Short(_FakeProvider):
        def pipe(
            self, texts: Iterable[str], *, batch_size: int
        ) -> Iterable[NLPDocument]:
            return iter(())

    register_nlp_provider(
        NLPProviderRegistration(
            name="short",
            loader=lambda spec: _Short("short", spec),
            capabilities=_ALL,
            languages=None,
        )
    )
    detector.update({"hello": "en"})

    with pytest.raises(ValueError, match="expected 1, got 0"):
        nlp_tokens.run(
            {"raw": _frame([("a", "hello", None)])},
            _ctx(_routing_options(routes={"en": {"provider": "short", "model": "m"}})),
        )


def test_unrouted_transforms_keep_their_output_schema() -> None:
    _register("fake")
    upstream = pl.DataFrame({"document_id": ["a"], "text": ["hello"]})

    tokens = nlp_tokens.run({"raw": upstream}, _ctx({"provider": "fake"}))

    assert not {"nlp_route", "nlp_input_language", "nlp_language_source"} & set(
        tokens.columns
    )


def test_the_documented_routing_example_validates() -> None:
    """The block in docs/reference.md, so the docs cannot drift from the models."""
    options = {
        "document_id_field": "document_id",
        "text_field": "text",
        "language_routing": {
            "language_field": "lang",
            "routes": {
                "en": {"provider": "spacy", "model": "en_core_web_sm"},
                "de": {"provider": "spacy", "model": "de_core_news_sm"},
                "zh-cn": {
                    "provider": "spacy",
                    "model": "zh_core_web_sm",
                    "language": "zh",
                },
            },
            "on_undetected": "skip",
            "on_unrouted": "fallback",
            "fallback": {
                "provider": "spacy",
                "model": "xx_ent_wiki_sm",
                "language": "xx",
            },
        },
    }

    parsed = NLPTokenOptions.model_validate(options)
    nlp_entities.validate_options(options)

    assert parsed.language_routing is not None
    assert parsed.language_routing.route_language("zh-cn") == "zh"
    assert nlp_tokens.declared_incremental_contract(options).parent_key == "document_id"
