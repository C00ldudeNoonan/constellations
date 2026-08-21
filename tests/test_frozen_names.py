"""Pins for the strings whose value is data rather than code.

Some names stel uses are written into places that outlive a single run: a
warehouse, a LanceDB collection, a fitted artifact on disk, an emitted dbt
project, an operator's environment. Changing one of those does not break a
build — it points the tool at something that is not there. The next run reports
every document as new and reprocesses the corpus at full provider cost, green
the whole way.

Nothing else in the suite catches that. `test_hashing.py` asserts only relative
properties (same input, same digest), which hold no matter what the fingerprint
prefix is, so today `_FINGERPRINT_PREFIX` could change and every test would
still pass while every incremental cache silently invalidated. Hence real
digests below rather than properties.

A deliberate contract change must update the tables here, and that conscious
update is the point: the `why_frozen` text is carried into the assertion
message so a failure explains what breaks rather than just which byte moved.
See #313.

#313 is also the one time this happened on purpose. The internal warehouse
table names moved from `dbt_ml_*` to `stel_*`, carried by `stel migrate` and
fenced by the connect-time guard in `WarehouseAdapter._guard_legacy_names`.
Both spellings are pinned below: the current one because it is what live
warehouses hold now, and the `LEGACY_*` one because it is what pre-rename
warehouses still hold and what the migration and the guards look for. Losing
a legacy value would leave that data unreachable and unmentioned.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import polars as pl
import pytest

import stel
from stel import _distribution, hashing, logging_setup
from stel import env as env_module
from stel.adapters import (
    StateRecord,
    StateScope,
    create_adapter,
    parse_warehouse_config,
)
from stel.adapters.base import (
    LEGACY_SERVING_LEASE_TABLE,
    LEGACY_SERVING_LEDGER_TABLE,
    LEGACY_STAGING_TABLE_PREFIX,
    LEGACY_STATE_TABLE,
    LEGACY_TEST_FAILURES_TABLE_PREFIX,
    SERVING_LEASE_TABLE,
    SERVING_LEDGER_TABLE,
    STAGING_TABLE_PREFIX,
    STATE_TABLE,
    TEST_FAILURES_TABLE_PREFIX,
    staging_table_name,
)
from stel.checks.schema import _failures_table_name
from stel.classic_ml.artifacts import ARTIFACT_RUNTIME_VERSION_KEY
from stel.config.identifiers import (
    DEFAULT_DUCKDB_FILENAME,
    DEFAULT_SCHEMA_NAME,
    GLOBAL_PROFILES_DIRNAME,
    LEGACY_DUCKDB_FILENAME,
    LEGACY_GLOBAL_PROFILES_DIRNAME,
    LEGACY_PROJECT_FILENAME,
    LEGACY_SCHEMA_NAME,
    PROJECT_FILENAME,
    RESERVED_PREFIXES,
)
from stel.config.profile import WarehouseConfig
from stel.dbt_export import (
    DBT_META_NAMESPACE,
    DEFAULT_SOURCE_NAME_PREFIX,
    default_dbt_source_name,
)
from stel.execution import extraction as extraction_module
from stel.manifest import RUN_RESULTS_VERSION_KEY
from stel.providers import base as provider_base  # registers providers
from stel.providers.anthropic import AnthropicInferenceProvider
from stel.providers.base import BaseProvider, implementation_identity_for
from stel.providers.deterministic import (
    DeterministicEmbeddingProvider,
    DeterministicInferenceProvider,
)
from stel.providers.vertex import VertexEmbeddingProvider, VertexInferenceProvider
from stel.providers.vllm import VLLMInferenceProvider
from stel.retrieval import coordination, lancedb

SRC_ROOT = pathlib.Path(stel.__file__).parent


# --- literal pins ----------------------------------------------------------

# (label, actual, expected, why_frozen)
_FROZEN_LITERALS: tuple[tuple[str, object, object, str], ...] = (
    (
        "adapters.base.STATE_TABLE",
        STATE_TABLE,
        "stel_state",
        "holds every incremental fingerprint; a new name is an empty state table, "
        "so every model reprocesses its whole corpus at provider cost",
    ),
    (
        "adapters.base.LEGACY_STATE_TABLE",
        LEGACY_STATE_TABLE,
        "dbt_ml_state",
        "the pre-#313 spelling still present in warehouses built before the "
        "rename; `stel migrate` and the connect guard find that state by this "
        "name and lose it if the value drifts",
    ),
    (
        "adapters.base.SERVING_LEDGER_TABLE",
        SERVING_LEDGER_TABLE,
        "stel_serving_ledger",
        "fenced state replacement verifies publication claims against this table; "
        "a new name strands every live claim",
    ),
    (
        "adapters.base.LEGACY_SERVING_LEDGER_TABLE",
        LEGACY_SERVING_LEDGER_TABLE,
        "dbt_ml_serving_ledger",
        "the pre-#313 spelling `stel migrate` carries over; a drift here strands "
        "the claims it was supposed to move",
    ),
    (
        "adapters.base.SERVING_LEASE_TABLE",
        SERVING_LEASE_TABLE,
        "stel_serving_leases",
        "a new name strands the live leases and publishers lose sight of who "
        "holds what",
    ),
    (
        "adapters.base.LEGACY_SERVING_LEASE_TABLE",
        LEGACY_SERVING_LEASE_TABLE,
        "dbt_ml_serving_leases",
        "the pre-#313 spelling `stel migrate` carries over; a drift here strands "
        "the leases it was supposed to move",
    ),
    (
        "retrieval.coordination.LEASE_TABLE",
        coordination.LEASE_TABLE,
        SERVING_LEASE_TABLE,
        "coordination owns the lease protocol but not its name: migration "
        "planning lives in adapters and cannot import retrieval to learn it",
    ),
    (
        "adapters.base.TEST_FAILURES_TABLE_PREFIX",
        TEST_FAILURES_TABLE_PREFIX,
        "stel_test_failures__",
        "list_tables() hides this prefix; a producer/filter mismatch leaks "
        "--store-failures tables into the model namespace",
    ),
    (
        "adapters.base.STAGING_TABLE_PREFIX",
        STAGING_TABLE_PREFIX,
        "stel_staging__",
        "list_tables() hides this prefix; a producer/filter mismatch leaks "
        "in-flight load tables into the model namespace",
    ),
    (
        "adapters.base.LEGACY_TEST_FAILURES_TABLE_PREFIX",
        LEGACY_TEST_FAILURES_TABLE_PREFIX,
        "dbt_ml_test_failures__",
        "still hidden by list_tables() and renamed by `stel migrate`; dropping "
        "it surfaces pre-rename internals as if a user had modeled them",
    ),
    (
        "adapters.base.LEGACY_STAGING_TABLE_PREFIX",
        LEGACY_STAGING_TABLE_PREFIX,
        "dbt_ml_staging__",
        "still hidden by list_tables(); orphaned pre-rename staging tables from "
        "a crashed run would otherwise appear in every catalog listing",
    ),
    (
        "config.identifiers.RESERVED_PREFIXES",
        RESERVED_PREFIXES,
        ("dbt_ml_", "stel_"),
        "existing projects rely on dbt_ml_ staying rejected even though the "
        "internals moved off it, and reserving a prefix newly forbids model "
        "names that were legal before, so this list only ever grows",
    ),
    (
        "config.identifiers.DEFAULT_SCHEMA_NAME",
        DEFAULT_SCHEMA_NAME,
        "stel",
        "holds every table a project materializes; a deployment that never set "
        "`schema:` explicitly loses sight of all of them at once",
    ),
    (
        "config.identifiers.LEGACY_SCHEMA_NAME",
        LEGACY_SCHEMA_NAME,
        "dbt_ml",
        "the schema pre-#313 deployments defaulted into; the connect guard reads "
        "it to tell a fresh project apart from one pointed at the wrong schema",
    ),
    (
        "config.identifiers.DEFAULT_DUCKDB_FILENAME",
        DEFAULT_DUCKDB_FILENAME,
        "stel.duckdb",
        "a new default silently points the legacy no-profile path at an empty "
        "database",
    ),
    (
        "config.identifiers.LEGACY_DUCKDB_FILENAME",
        LEGACY_DUCKDB_FILENAME,
        "dbt_ml.duckdb",
        "the file pre-#313 zero-config projects defaulted to; config load looks "
        "for it before opening an empty database under the new name",
    ),
    (
        "config.identifiers.PROJECT_FILENAME",
        PROJECT_FILENAME,
        "stel_project.yml",
        "every existing project has this file on disk under this exact name",
    ),
    (
        "config.identifiers.LEGACY_PROJECT_FILENAME",
        LEGACY_PROJECT_FILENAME,
        "dbt_ml_project.yml",
        "what a pre-rename project still has on disk, and the only reason the "
        "missing-file error can explain itself instead of just reporting an "
        "absence",
    ),
    (
        "config.identifiers.GLOBAL_PROFILES_DIRNAME",
        GLOBAL_PROFILES_DIRNAME,
        ".stel",
        "the global profile lives here; a new name silently finds no profile",
    ),
    (
        "config.identifiers.LEGACY_GLOBAL_PROFILES_DIRNAME",
        LEGACY_GLOBAL_PROFILES_DIRNAME,
        ".dbt_ml",
        "where a pre-rename global profile still sits, and what the not-found "
        "error points at",
    ),
    (
        "_distribution.DISTRIBUTION_NAME",
        _distribution.DISTRIBUTION_NAME,
        "stel",
        "must equal [project] name in pyproject.toml; a mismatch does not raise, "
        "it reports the version as 'unknown' everywhere",
    ),
    (
        "manifest.RUN_RESULTS_VERSION_KEY",
        RUN_RESULTS_VERSION_KEY,
        "dbt_ml_version",
        "run_results metadata is the payload Dagster reads (#87)",
    ),
    (
        "dbt_export.DBT_META_NAMESPACE",
        DBT_META_NAMESPACE,
        "dbt_ml",
        "the emitted sources.yml is committed into the consumer's dbt project and "
        "read from there",
    ),
    (
        "dbt_export.DEFAULT_SOURCE_NAME_PREFIX",
        DEFAULT_SOURCE_NAME_PREFIX,
        "dbt_ml_",
        "the emitted `sources:` name is committed into the consumer's dbt "
        "project and named by every `source()` call in it, so a new default "
        "breaks their models rather than ours",
    ),
    (
        "classic_ml.artifacts.ARTIFACT_RUNTIME_VERSION_KEY",
        ARTIFACT_RUNTIME_VERSION_KEY,
        "dbt_ml",
        "required and validated on load, so a new key fails every fitted artifact "
        "already on disk",
    ),
    (
        "hashing._FINGERPRINT_PREFIX",
        hashing._FINGERPRINT_PREFIX,
        b"dbt-ml-canonical-fingerprint",
        "mixed into every fingerprint; one byte different invalidates every "
        "digest ever stored",
    ),
    (
        "retrieval.lancedb._OWNER_KEY",
        lancedb._OWNER_KEY,
        b"dbt_ml.owner",
        "Arrow schema metadata key on every published collection",
    ),
    (
        "retrieval.lancedb._OWNER_VALUE",
        lancedb._OWNER_VALUE,
        b"dbt-ml",
        "compared exactly when adopting a collection, so it gates reads rather "
        "than labelling them: a new value makes every published collection "
        "unreadable",
    ),
    (
        "retrieval.lancedb._CONTRACT_KEY",
        lancedb._CONTRACT_KEY,
        b"dbt_ml.record_contract",
        "Arrow schema metadata key on every published collection",
    ),
    (
        "retrieval.lancedb._CONFIG_KEY",
        lancedb._CONFIG_KEY,
        b"dbt_ml.config_fingerprint",
        "Arrow schema metadata key on every published collection",
    ),
    (
        "retrieval.lancedb._IMPLEMENTATION_IDENTITY_PREFIX",
        lancedb._IMPLEMENTATION_IDENTITY_PREFIX,
        "dbt_ml.retrieval.lancedb:v1",
        "recorded on published search indexes and compared on read, so a new "
        "value invalidates published state",
    ),
    (
        "execution.extraction._FETCH_DIR_PREFIX",
        extraction_module._FETCH_DIR_PREFIX,
        "dbt_ml_fetch_",
        "the #273 startup sweep only ever sees directories a dead process left "
        "behind, so a producer/filter mismatch disables the self-healing silently",
    ),
    (
        "execution.extraction._OWNER_MARKER_NAME",
        extraction_module._OWNER_MARKER_NAME,
        ".dbt_ml_owner_pid",
        "distinguishes a live run's staging directory from an abandoned one",
    ),
    ("env.PROFILES_DIR_ENV", env_module.PROFILES_DIR_ENV, "STEL_PROFILES_DIR", "operator-set"),
    ("env.VERBOSE_ENV", env_module.VERBOSE_ENV, "STEL_VERBOSE", "operator-set"),
    (
        "env.PROVIDER_DEBUG_ENV",
        env_module.PROVIDER_DEBUG_ENV,
        "STEL_DEBUG_PROVIDER_ERRORS",
        "operator-set",
    ),
    (
        "env.MCP_PRINCIPAL_ID_ENV",
        env_module.MCP_PRINCIPAL_ID_ENV,
        "STEL_MCP_PRINCIPAL_ID",
        "operator-set; when it stops resolving the MCP boundary builds no "
        "principal at all rather than erroring",
    ),
    (
        "env.MCP_TENANT_ID_ENV",
        env_module.MCP_TENANT_ID_ENV,
        "STEL_MCP_TENANT_ID",
        "operator-set; when it stops resolving the principal is built with no "
        "tenant rather than erroring",
    ),
    (
        "env.MCP_ACCESS_GROUPS_ENV",
        env_module.MCP_ACCESS_GROUPS_ENV,
        "STEL_MCP_ACCESS_GROUPS",
        "operator-set; when it stops resolving the principal is built with no "
        "access groups rather than erroring",
    ),
    (
        "env.MCP_POLICY_CLAIMS_ENV",
        env_module.MCP_POLICY_CLAIMS_ENV,
        "STEL_MCP_POLICY_CLAIMS",
        "operator-set; when it stops resolving the principal is built with no "
        "policy claims rather than erroring",
    ),
)


@pytest.mark.parametrize(
    ("label", "actual", "expected", "why_frozen"),
    _FROZEN_LITERALS,
    ids=[row[0] for row in _FROZEN_LITERALS],
)
def test_frozen_literal(label: str, actual: object, expected: object, why_frozen: str) -> None:
    assert actual == expected, (
        f"{label} changed from {expected!r} to {actual!r}. This value is frozen: "
        f"{why_frozen}. Changing it needs a migration that carries the existing "
        "data over, not an edit here."
    )


def test_verbose_logger_namespace_matches_the_package() -> None:
    # A handler attached to a namespace no module logs under silences `-v`
    # without failing anything.
    assert logging_setup._ROOT_LOGGER == stel.__name__


def test_resolved_version_is_not_unknown() -> None:
    # `distribution_version()` degrades to "unknown" instead of raising, so a
    # distribution name that no longer matches pyproject.toml is invisible.
    assert stel.__version__ != _distribution.UNKNOWN_VERSION
    assert _distribution.distribution_version() != _distribution.UNKNOWN_VERSION


def test_type_identity_is_qualified_by_module() -> None:
    # Fingerprinting a value that carries a stel-defined pydantic model or
    # enum records its module path in the digest. No production call site does
    # that today, but this pin makes the coupling visible if the package is
    # ever renamed.
    assert (
        hashing._type_identity(WarehouseConfig()) == "stel.config.profile.WarehouseConfig"
    )


# --- provider identity -----------------------------------------------------

# Provider implementation identity keys cached provider responses and the state
# of every `llm:`/`embed:` model. It deliberately excludes the release version
# and module source digests so those caches survive an upgrade — which makes it
# the one identity a package rename must not move. Digests captured before the
# #313 rename; `_identity_qualname` is what keeps them.
_PROVIDER_IDENTITIES: tuple[tuple[str, str], ...] = (
    ("anthropic-inference", "provider-v3/5c08808d1ce2631b38df9019744a80fe"),
    ("deterministic-embedding", "provider-v3/525af5c1b28b0dd4e9fd2038b89315d2"),
    ("deterministic-inference", "provider-v3/5ebfdb5a1ba32560062b0d962c86f057"),
    ("vertex-embedding", "provider-v3/da9e3b794607ec572d78ce1954240d90"),
    ("vertex-inference", "provider-v3/9e1c7a01973ce064223209be19ccf7a9"),
    ("vllm-inference", "provider-v3/091d89d921dfbd7116acfa1d1c962785"),
)


def test_provider_identities_survive_the_package_rename() -> None:
    # Enumerated explicitly rather than by walking __subclasses__: other test
    # modules define their own providers, so a discovery-based list depends on
    # import order.
    providers: dict[str, type[BaseProvider]] = {
        "anthropic-inference": AnthropicInferenceProvider,
        "deterministic-embedding": DeterministicEmbeddingProvider,
        "deterministic-inference": DeterministicInferenceProvider,
        "vertex-embedding": VertexEmbeddingProvider,
        "vertex-inference": VertexInferenceProvider,
        "vllm-inference": VLLMInferenceProvider,
    }
    pinned = dict(_PROVIDER_IDENTITIES)
    assert set(providers) == set(pinned)

    # Other tests monkeypatch the dependency-version lookup this identity reads.
    provider_base._implementation_identity.cache_clear()
    actual = {label: implementation_identity_for(cls) for label, cls in providers.items()}

    assert actual == pinned, (
        "A provider implementation identity changed. That re-keys every cached "
        "provider response and the state of every llm:/embed: model, so the next "
        "run re-calls the provider for rows it already has. The identity excludes "
        "the release version on purpose; if this failed after a module move, see "
        "providers.base._identity_qualname."
    )


# --- fingerprint goldens ---------------------------------------------------

# A fixed payload spanning the canonical encoder's scalar and container cases.
# Never change this: its whole purpose is that the digests below stay comparable.
_PROBE = {
    "text": "probe",
    "count": 7,
    "ratio": 0.5,
    "flag": True,
    "empty": None,
    "items": [1, "two"],
    "nested": {"key": "value"},
}

# (domain, version, digest of _PROBE). Every `canonical_fingerprint` call site
# in src/ is represented; test_every_fingerprint_domain_is_pinned enforces that.
_FINGERPRINT_GOLDENS: tuple[tuple[str, int, str], ...] = (
    ("chunk-input", 2, "94c9f5e4a79220f0d0db180a36783470"),
    ("dbt-ml-agent-context-document", 1, "9a7d138624cf31cbd52ea165510dca30"),
    ("dbt-ml-agent-context-document-version", 1, "f14d8f4b9acb0001224fbbd22286c57c"),
    ("dbt-ml-agent-context-entity", 1, "3e26c34b20f0dcd611feeb3c98a69762"),
    ("dbt-ml-agent-context-entity-link", 1, "72d2f5e8f67c079e4b1e9a258b7841c5"),
    ("dbt-ml-agent-context-policy", 1, "91be9ce477d543dd69a1151e7bb89f9a"),
    ("dbt-ml-agent-context-provenance", 1, "4ca78cd7a241eee1e26d544d9880d36b"),
    ("dbt-ml-agent-context-record", 1, "8f212cefa1a8dc90c38a61ba0de0a83b"),
    ("dbt-ml-agent-context-retrieval-projection", 1, "5f960ba22b6922d177ea64691527168e"),
    ("dbt-ml-bigquery-table-generation", 1, "4c2d179040034d4f0f2ad598f089c6ad"),
    ("dbt-ml-lancedb-generation", 1, "e5ff0b2eaebe595840ffe559abb5e143"),
    ("dbt-ml-provider-profile-options", 1, "a2a7017c8e3463ba9e4b09cb93f62272"),
    ("dbt-ml-safe-retrieval-target", 1, "96f83dc773df6a841603531422381937"),
    ("dbt-ml-safe-warehouse-target", 1, "0598cd98347e5409e38c014cb61aba43"),
    ("dbt-ml-search-collection-config", 1, "04352c92c1c98c882ccbe965824064bf"),
    ("dbt-ml-search-delete-batch", 1, "83e10ac440b5c4ac94829907c7e9812d"),
    ("dbt-ml-search-governed-revoke-batch", 1, "b68b27e0446fa6058104832629ff7a99"),
    ("dbt-ml-search-indexed-row", 1, "27914404645f6e521b77f258f1bb1300"),
    ("dbt-ml-search-upsert-batch", 1, "ca587322c28f29eae4bb9c2b102c4ddb"),
    ("dbt-ml-state-target-identity", 1, "f087496846b3017f361a79ae14ffb95a"),
    ("dbt-ml-warehouse-table-generation", 1, "6d5a58592fe52d34fb887b2bc9ade896"),
    ("dbt-ml-warehouse-table-snapshot", 1, "fe0f9290da11e207f1aba25686262d87"),
    ("dbt-ml.entity-alias-set", 1, "4a8fbb1a4062e79cb9350615547695ab"),
    ("dbt-ml.entity-link", 1, "8c04bf8fb62886b5e9acbcace10667df"),
    ("dbt-ml.entity-relation", 1, "f16b110f060f1acbd5eca3b299859d8f"),
    ("dbt-ml.entity-vector-reference-set", 1, "31aa261aa58ee62e0a07a7e7fbe7fd7d"),
    ("dbt-ml.keyphrase", 1, "26e674c01ac3e998d74e2200dc9018bc"),
    ("dbt-ml.nlp-entity", 1, "1210a385cadbdd8b52560f4e909afab9"),
    ("dbt-ml.nlp-token", 1, "3353cd8ad6c7715d403d7a94f7a386be"),
    ("dbt-ml.tone-lexicon", 1, "e1ba3829a480e432af1393119eb0ff81"),
    ("dbt-ml.transform-incremental-input", 1, "c83fb706e8ba9060850314bcb687c9c6"),
    ("dbt-ml.transform-incremental-reference", 1, "9f69628fa9a3e1f668ec869219282eb0"),
    ("dbt_ml/retrieval_eval/golden_set", 1, "9305ca078ff6376cce65b750ebaf21cf"),
    ("eval-metric-id", 1, "a857cb0a824c56aa011e6b9b88cbba6e"),
    ("warehouse-source-row", 1, "362b1a57365cd242b114e1e95a647e79"),
    ("embedding-config", 1, "a1f45bc702e4875dad7d01e066f92374"),
    ("embedding-input-row", 1, "4c7f5bfe371485b0aa73c4b777011d4f"),
    ("embedding-input-text", 1, "d3bff0761e0d18692fd51fa87dc379cd"),
    ("llm-input-content", 1, "03c6810c0553f536003b67bb80e3f124"),
    ("llm-map-config", 1, "5ed9c5ffcb8d96f026b8a3bbd2946510"),
)


@pytest.mark.parametrize(
    ("domain", "version", "digest"),
    _FINGERPRINT_GOLDENS,
    ids=[f"{domain}-v{version}" for domain, version, _ in _FINGERPRINT_GOLDENS],
)
def test_fingerprint_golden(domain: str, version: int, digest: str) -> None:
    actual = hashing.canonical_fingerprint(_PROBE, domain=domain, version=version)
    assert actual == digest, (
        f"The fingerprint for domain {domain!r} v{version} changed. Every digest "
        "already stored under it — incremental state, cache keys, agent-context "
        "ids — no longer matches, so the next run reprocesses everything at "
        "provider cost and reports success. Either the domain string, the "
        "fingerprint prefix, or the canonical encoder changed."
    )


# --- completeness scans over src/ -----------------------------------------


def _source_files() -> list[pathlib.Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def _module_name(path: pathlib.Path) -> str:
    relative = path.relative_to(SRC_ROOT.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _declared_fingerprint_domains() -> set[tuple[str, int]]:
    """Every (domain, version) pair `canonical_fingerprint` is called with."""
    declared: set[tuple[str, int]] = set()
    for path in _source_files():
        tree = ast.parse(path.read_text())
        module = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _called_name(node) != "canonical_fingerprint":
                continue
            keywords = {kw.arg: kw.value for kw in node.keywords}
            domain_node = keywords.get("domain")
            assert domain_node is not None, (
                f"{path}:{node.lineno} calls canonical_fingerprint without a "
                "keyword `domain=`; this scan cannot see the domain, so it "
                "cannot be pinned"
            )
            if isinstance(domain_node, ast.Constant):
                domain = domain_node.value
            elif isinstance(domain_node, ast.Name):
                # A module-level constant, possibly imported. Resolving through
                # the imported module namespace covers both cases.
                if module is None:
                    module = importlib.import_module(_module_name(path))
                domain = getattr(module, domain_node.id)
            else:
                raise AssertionError(
                    f"{path}:{node.lineno} passes a `domain=` this scan cannot "
                    f"resolve ({ast.dump(domain_node)}). Pass a literal or a "
                    "module-level constant so the domain stays pinnable."
                )
            version_node = keywords.get("version")
            version = 1 if version_node is None else ast.literal_eval(version_node)
            assert isinstance(domain, str) and isinstance(version, int)
            declared.add((domain, version))
    return declared


def test_every_fingerprint_domain_is_pinned() -> None:
    declared = _declared_fingerprint_domains()
    pinned = {(domain, version) for domain, version, _ in _FINGERPRINT_GOLDENS}

    assert declared == pinned, (
        "The set of fingerprint domains in src/ no longer matches the pinned "
        f"goldens.\n  unpinned: {sorted(declared - pinned)}\n  stale pins: "
        f"{sorted(pinned - declared)}\nA new domain needs a golden digest here; "
        "a removed one means the digests stored under it are orphaned."
    )


_ENV_READERS = frozenset({"getenv", "read_env"})


def _env_read_arguments(node: ast.AST) -> list[ast.expr]:
    """The name arguments of an environment read, if `node` is one."""
    if isinstance(node, ast.Call):
        name = _called_name(node)
        if name in _ENV_READERS:
            return list(node.args)
        if (
            name == "get"
            and isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func.value).endswith("environ")
        ):
            return list(node.args)
    elif isinstance(node, ast.Subscript) and ast.unparse(node.value).endswith("environ"):
        return [node.slice]
    return []


def _env_literals() -> list[str]:
    """Bare `STEL*` string literals passed to an environment read."""
    offenders: list[str] = []
    for path in _source_files():
        for node in ast.walk(ast.parse(path.read_text())):
            for argument in _env_read_arguments(node):
                if (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and argument.value.startswith("STEL")
                ):
                    offenders.append(f"{path.name}:{argument.lineno} {argument.value}")
    return offenders


def test_no_bare_env_var_literals() -> None:
    offenders = _env_literals()
    assert not offenders, (
        "Environment variables must be read through the constants in stel.env "
        "so the set stays enumerable and a rename is one diff: "
        f"{offenders}"
    )


def test_env_scan_recognizes_every_form_the_code_could_regress_to() -> None:
    # `test_no_bare_env_var_literals` passes trivially if the matching is
    # broken, so prove the scan sees each way the codebase used to read a
    # variable — including the two forms it had before #313.
    source = (
        "import os\n"
        'a = os.environ.get("STEL_VERBOSE", "")\n'
        'b = os.environ["STEL_VERBOSE"]\n'
        'c = os.getenv("STEL_VERBOSE")\n'
        'd = read_env("STEL_VERBOSE")\n'
    )
    matched = [
        ast.literal_eval(argument)
        for node in ast.walk(ast.parse(source))
        for argument in _env_read_arguments(node)
    ]
    assert matched.count("STEL_VERBOSE") == 4


def test_every_producer_of_the_dbt_source_name_uses_the_shared_helper() -> None:
    """The default source name was spelled inline at three sites, and one of
    them (`concept_cloud.export`) reconstructed it in a way that ignored
    `--source-name` — a silent empty DAG join rather than an error. A pin on
    the value alone would not have caught that, because every copy agreed."""
    offenders: list[str] = []
    for path in _source_files():
        if path.name == "dbt_export.py":
            continue  # the owner
        text = path.read_text(encoding="utf-8")
        if 'f"dbt_ml_{' in text or "f'dbt_ml_{" in text:
            offenders.append(_module_name(path))
    assert offenders == [], (
        "These modules build the dbt source name themselves instead of calling "
        f"dbt_export.default_dbt_source_name: {offenders}. Every producer must "
        "go through the helper so a --source-name override reaches all of them."
    )


def test_the_helper_still_produces_the_frozen_name() -> None:
    assert default_dbt_source_name("economic_data") == "dbt_ml_economic_data"


# --- lockstep behavior -----------------------------------------------------


def test_internal_table_producers_and_filter_agree(tmp_path: pathlib.Path) -> None:
    """Materialize each internal table the way production names it, then assert
    `list_tables()` hides it. This is what a shared constant is actually for:
    the pins above prove the values did not change, and this proves the producer
    and the filter still read the same one."""
    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "t.duckdb"), "schema": "frozen"}
    )
    frame = pl.DataFrame({"x": [1]})
    failures_table = _failures_table_name("model_a", "not_null", "x")
    staging_table = staging_table_name("model_a")

    with create_adapter(config) as adapter:
        adapter.materialize_full("model_a", frame)
        adapter.materialize_full(failures_table, frame)
        adapter.materialize_full(staging_table, frame)
        # The state table is created on demand by the state API, not by us.
        adapter.upsert_state(
            StateScope("model_a"),
            [StateRecord("doc-1", "fingerprint", "code-v1")],
        )

        # The serving ledger and leases are created by retrieval.coordination,
        # which needs a whole publication to run. Their names are what is under
        # test here, so materialize them directly.
        adapter.materialize_full(SERVING_LEDGER_TABLE, frame)
        adapter.materialize_full(SERVING_LEASE_TABLE, frame)
        # Debris a pre-#313 run could have left behind. list_tables() has to go
        # on hiding it, or upgrading surfaces old internals as if they were
        # models a user had written.
        adapter.materialize_full(LEGACY_STATE_TABLE, frame)
        adapter.materialize_full(LEGACY_TEST_FAILURES_TABLE_PREFIX + "old", frame)
        adapter.materialize_full(LEGACY_STAGING_TABLE_PREFIX + "old__abc", frame)

        physical = set(adapter.list_all_tables())
        # Guard against the test passing because nothing was created.
        assert {"model_a", failures_table, staging_table, STATE_TABLE} <= physical

        assert adapter.list_tables() == ["model_a"]
        assert failures_table.startswith(TEST_FAILURES_TABLE_PREFIX)
        assert staging_table.startswith(STAGING_TABLE_PREFIX)


# The keys of the payload `BaseBackend.implementation_identity` canonicalizes,
# in the order they must hash to. `json.dumps(..., sort_keys=True)` means the
# key spellings are themselves hashed, so this tuple is data: renaming a key,
# adding one, or dropping one re-keys every backend identity at once, with no
# behavior change to explain it.
#
# The blast radius is re-extraction, not automatically re-spend: `_cache_key`
# in backends/llm_backend.py is keyed on the *provider* contract identity
# rather than the backend, precisely so cached responses survive routine
# upgrades. So a re-key costs wall-clock against a warm response cache, and
# real provider spend only where that cache is cold or absent (a fresh
# machine, CI, a non-LLM backend has no cache to miss). Still not something to
# spend by accident, and nothing else in the suite notices: the identity test
# in test_backend_options.py asserts only distinctness and the `stel/` prefix,
# both of which survive a rename.
_BACKEND_IDENTITY_PAYLOAD_KEYS = (
    "backend_class",
    "backend_class_source",
    "backend_module_source",
    "base_source",
    "dbt_ml_version",
)


def _backend_identity_payload_keys() -> tuple[str, ...]:
    """The payload keys, read from the source rather than by calling it.

    `implementation_identity()` returns a digest, so calling it cannot show
    which keys went in — and a golden digest would be the wrong pin, because
    the payload carries source digests that are *supposed* to move whenever
    backend code changes. The key names are the part that must not.
    """
    tree = ast.parse((SRC_ROOT / "backends" / "base.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "implementation_identity":
            continue
        for statement in ast.walk(node):
            if not isinstance(statement, ast.Assign):
                continue
            target = statement.targets[0]
            if not isinstance(target, ast.Name) or target.id != "payload":
                continue
            assert isinstance(statement.value, ast.Dict), (
                "implementation_identity's `payload` is no longer a dict "
                "literal; this scan can no longer see the keys that are hashed"
            )
            keys = []
            for key in statement.value.keys:
                assert isinstance(key, ast.Constant) and isinstance(key.value, str), (
                    f"implementation_identity payload has a non-literal key "
                    f"({ast.dump(key) if key else 'None'}); keep them literals "
                    "so they stay pinnable"
                )
                keys.append(key.value)
            return tuple(sorted(keys))
    raise AssertionError(
        "BaseBackend.implementation_identity was not found in backends/base.py"
    )


def test_backend_identity_payload_keys_are_frozen() -> None:
    assert _backend_identity_payload_keys() == _BACKEND_IDENTITY_PAYLOAD_KEYS, (
        "The backend implementation-identity payload keys changed. Those keys "
        "are hashed (sort_keys=True), and the digest gates incremental "
        "invalidation for every `extraction:` model — so this change reports "
        "every document as new and re-extracts every corpus, green the whole "
        "way. Cached provider responses survive (they key on the provider "
        "contract identity, not the backend), so the bill is wall-clock on a "
        "warm cache and real provider spend on a cold one. `dbt_ml_version` "
        "in particular keeps its pre-#313 spelling on purpose, for the same "
        "reason `providers.base._IDENTITY_PACKAGE` does."
    )
