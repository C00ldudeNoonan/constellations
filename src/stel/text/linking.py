"""Entity-linking resolver contracts, registry, and the built-in resolvers.

Three resolvers ship today, all deterministic and offline:

- ``alias_table`` joins mention text to an alias dimension with exact and
  normalized text matching. It never guesses: every mention outcome is an
  explicit ``matched``, ``ambiguous``, or ``unmatched`` status, and ambiguous
  candidates are preserved as separate rows. The alias dimension is either an
  operator-owned upstream model, or — for ``aliases: vocab.<name>`` — a
  project-declared vocabulary's preferred labels and alternative labels
  (issue #627), read directly: no table to materialize or keep in sync.
- ``vector_similarity`` matches precomputed mention embeddings against
  precomputed alias embeddings by cosine/dot/euclidean similarity above a
  threshold. Both vectors are produced upstream by the ``embed`` model kind, so
  credentials, provider batching, and versioned embedding identity stay in that
  executor and this resolver remains a pure offline transform over frame data.
- ``fuzzy`` matches alias text past exact/normalized matching — character-trigram
  Dice or whitespace-token Jaccard — for spelling variants, legal suffixes, and
  typos ``alias_table`` would miss.
"""
from __future__ import annotations

import math
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from ..config.vocabulary import Vocabulary
from ..entity_link_options import (
    AliasTableResolverOptions,
    EntityLinkBaseOptions,
    FuzzyMetric,
    FuzzyResolverOptions,
    MatchMethod,
    SimilarityMetric,
    VectorSimilarityResolverOptions,
)
from ..hashing import canonical_fingerprint

# Bumped whenever a resolver's matching semantics change (normalization rules,
# method precedence, status assignment, similarity math) so downstream consumers
# can invalidate rows produced by an older resolver even though the package
# version moved for unrelated reasons.
#
# 2: `alias_table`'s `aliases:` option may source its alias rows from a
# declared vocabulary (`vocab.<name>`) instead of an upstream model (issue
# #627) — a new input shape, so existing cached/materialized rows must not be
# read as produced under it.
ALIAS_RESOLVER_VERSION = "2"
VECTOR_SIMILARITY_RESOLVER_VERSION = "1"
FUZZY_RESOLVER_VERSION = "1"

# Frozen fingerprint domains: mixed into persisted digests, so a change here
# invalidates every stored one. See hashing._FINGERPRINT_PREFIX.
ALIAS_SET_FINGERPRINT_DOMAIN = "dbt-ml.entity-alias-set"
VECTOR_REFERENCE_FINGERPRINT_DOMAIN = "dbt-ml.entity-vector-reference-set"
ENTITY_LINK_FINGERPRINT_DOMAIN = "dbt-ml.entity-link"

LinkStatus = Literal["matched", "ambiguous", "unmatched"]


# --- Resolver contract -------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    canonical_id: str
    score: float | None


@dataclass(frozen=True)
class NamespaceResolution:
    """One namespace's outcome for a mention. ``status`` is ``matched`` for a
    single winner or ``ambiguous`` when several candidates are indistinguishable;
    a namespace with no acceptable candidate is simply omitted (the driver emits
    ``unmatched`` when a mention resolves no namespace at all)."""

    method: str
    status: Literal["matched", "ambiguous"]
    candidates: tuple[Candidate, ...]


def _resolve_scored_candidates(
    scored_by_namespace: Mapping[str, list[tuple[float, str]]],
    *,
    margin: float,
    method: str,
) -> dict[str, NamespaceResolution]:
    """Turn per-namespace ``(score, canonical_id)`` candidates that already
    cleared the acceptance threshold into resolutions. Within a namespace the
    top score wins; any candidate within ``margin`` of it is preserved as an
    equally-plausible winner, so several winners yield ``ambiguous`` rather than
    a silently arg-maxed guess. Shared by every score-producing resolver."""
    resolved: dict[str, NamespaceResolution] = {}
    for namespace, scored in scored_by_namespace.items():
        top = max(score for score, _ in scored)
        winners = sorted({cid for score, cid in scored if top - score <= margin})
        best_score = {
            cid: max(score for score, other in scored if other == cid)
            for cid in winners
        }
        status: Literal["matched", "ambiguous"] = (
            "matched" if len(winners) == 1 else "ambiguous"
        )
        resolved[namespace] = NamespaceResolution(
            method=method,
            status=status,
            candidates=tuple(
                Candidate(canonical_id=cid, score=best_score[cid]) for cid in winners
            ),
        )
    return resolved


class ResolverReference(ABC):
    """A resolver's validated, indexed reference data plus its fingerprint."""

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """The ``alias_set_version`` value: a one-way identity of the reference
        set so downstream consumers invalidate when it changes."""

    @abstractmethod
    def resolve(self, signal: Any) -> dict[str, NamespaceResolution]:
        """Resolve a prepared mention signal to per-namespace outcomes."""


class EntityResolver(ABC):
    name: str
    version: str

    def text_required(self) -> bool:
        """Whether the mention text column must be present regardless of
        ``include_mention_text`` (the alias-table resolver matches on text)."""
        return False

    @abstractmethod
    def required_mention_columns(self, options: Any) -> tuple[str, ...]:
        """Mention columns this resolver reads for its match signal."""

    @abstractmethod
    def mention_signal(self, row: Mapping[str, Any], options: Any) -> Any:
        """Extract and validate this mention's match signal, or return ``None``
        when there is nothing to match on (yielding an ``unmatched`` outcome)."""

    @abstractmethod
    def build_reference(self, frame: pl.DataFrame, options: Any) -> ResolverReference:
        """Validate and index the alias/reference frame."""

    def validate_frames(  # noqa: B027 - optional hook; default is intentionally a no-op
        self,
        mentions_frame: pl.DataFrame,
        aliases_frame: pl.DataFrame,
        options: Any,
    ) -> None:
        """Cross-frame preflight run before resolution. Default is a no-op;
        resolvers override it to reject inputs that would produce meaningless
        links (e.g. mention and alias vectors from different embedding spaces)."""


# --- Alias-table resolver ----------------------------------------------------


def vocabulary_alias_rows(vocab_name: str, vocabulary: Vocabulary) -> list[dict[str, str]]:
    """Alias-table rows for every term in a declared vocabulary.

    One `entity_namespace` per vocabulary — its own name — since `Vocabulary`
    has no separate namespace concept. Each term's own `label` is a row too,
    alongside its `aliases`, so a mention of the canonical name still matches:
    `labels()` keeps aliases out of what a classifier may *output*, but here
    the preferred label is exactly another string a mention may *say*.
    """
    rows: list[dict[str, str]] = []
    for term in vocabulary.terms:
        for alias in (term.label, *term.aliases):
            rows.append(
                {
                    "alias": alias,
                    "entity_namespace": vocab_name,
                    "canonical_id": term.label,
                }
            )
    return rows


def normalize_alias_text(value: str) -> str:
    """The alias resolver's ``normalized`` match key: NFKC fold, casefold, and
    whitespace collapse. Deliberately conservative — spelling variants and
    legal-suffix conventions belong in the operator-owned alias table."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def alias_set_fingerprint(rows: Iterable[Mapping[str, str]]) -> str:
    """One-way identity of the complete alias set (namespace, alias, canonical
    ID triples). Recorded on every link row so alias-table edits are visible to
    downstream invalidation without retaining the alias contents.

    Deduplicated so the fingerprint identifies the *effective* alias set that
    drives matching: repeated identical rows cannot change any link output, so
    they must not signal a spurious downstream invalidation."""
    canonical_rows = sorted(
        {(row["entity_namespace"], row["alias"], row["canonical_id"]) for row in rows}
    )
    return canonical_fingerprint(canonical_rows, domain=ALIAS_SET_FINGERPRINT_DOMAIN)


class _AliasTableReference(ResolverReference):
    def __init__(
        self,
        lookups: dict[str, dict[str, dict[str, set[str]]]],
        methods: tuple[MatchMethod, ...],
        fingerprint: str,
    ) -> None:
        self._lookups = lookups
        self._methods = methods
        self._fingerprint = fingerprint

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def resolve(self, signal: Any) -> dict[str, NamespaceResolution]:
        text = signal
        resolved: dict[str, NamespaceResolution] = {}
        if not isinstance(text, str) or not text.strip():
            return resolved
        # Methods run in configured order; the first producing candidates for a
        # namespace wins that namespace, while later methods may still contribute
        # other namespaces.
        for method in self._methods:
            key = text if method == "exact" else normalize_alias_text(text)
            hits = self._lookups[method].get(key)
            if not hits:
                continue
            for namespace, canonical_ids in hits.items():
                if namespace in resolved:
                    continue
                status: Literal["matched", "ambiguous"] = (
                    "matched" if len(canonical_ids) == 1 else "ambiguous"
                )
                resolved[namespace] = NamespaceResolution(
                    method=method,
                    status=status,
                    candidates=tuple(
                        Candidate(canonical_id=cid, score=None)
                        for cid in sorted(canonical_ids)
                    ),
                )
        return resolved


class AliasTableResolver(EntityResolver):
    name = "alias_table"
    version = ALIAS_RESOLVER_VERSION

    def text_required(self) -> bool:
        return True

    def required_mention_columns(self, options: Any) -> tuple[str, ...]:
        return ()

    def mention_signal(self, row: Mapping[str, Any], options: Any) -> Any:
        raw = row[options.mention_text_field]
        if raw is not None and not isinstance(raw, str):
            raise ValueError(
                f"Mention text column '{options.mention_text_field}' must contain "
                "strings or nulls"
            )
        return raw

    def build_reference(
        self, frame: pl.DataFrame, options: AliasTableResolverOptions
    ) -> _AliasTableReference:
        alias_rows = _reference_rows(
            frame,
            options,
            value_fields=(
                ("alias", options.alias_text_field),
                ("entity_namespace", options.namespace_field),
                ("canonical_id", options.canonical_id_field),
            ),
        )
        lookups: dict[str, dict[str, dict[str, set[str]]]] = {
            method: {} for method in options.match_methods
        }
        for row in alias_rows:
            for method, table in lookups.items():
                key = (
                    row["alias"]
                    if method == "exact"
                    else normalize_alias_text(row["alias"])
                )
                table.setdefault(key, {}).setdefault(
                    row["entity_namespace"], set()
                ).add(row["canonical_id"])
        return _AliasTableReference(
            lookups, options.match_methods, alias_set_fingerprint(alias_rows)
        )


# --- Vector-similarity resolver ----------------------------------------------


def _coerce_vector(value: Any, *, source: str, position: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(
            f"{source} vector at {position} must be a list of numbers"
        )
    vector: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(f"{source} vector at {position} must contain only numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{source} vector at {position} must be finite")
        vector.append(number)
    if not vector:
        return None
    return tuple(vector)


def _similarity(a: tuple[float, ...], b: tuple[float, ...], metric: SimilarityMetric) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    if metric == "dot":
        return dot
    if metric == "euclidean":
        # Negative distance so "higher is better" holds uniformly across metrics.
        return -math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class _AliasVector:
    namespace: str
    canonical_id: str
    vector: tuple[float, ...]


class _VectorReference(ResolverReference):
    def __init__(
        self,
        aliases: list[_AliasVector],
        dimensions: int,
        options: VectorSimilarityResolverOptions,
    ) -> None:
        self._aliases = aliases
        self._dimensions = dimensions
        self._metric = options.metric
        self._threshold = options.threshold
        self._margin = options.ambiguity_margin
        self._fingerprint = canonical_fingerprint(
            sorted(
                {(a.namespace, a.canonical_id, a.vector) for a in aliases}
            ),
            domain=VECTOR_REFERENCE_FINGERPRINT_DOMAIN,
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def resolve(self, signal: Any) -> dict[str, NamespaceResolution]:
        vector = signal
        if not isinstance(vector, tuple):
            return {}
        if not self._aliases:
            # A legitimately empty reference set has no dimensionality to enforce;
            # every mention resolves to no candidates rather than erroring.
            return {}
        if len(vector) != self._dimensions:
            raise ValueError(
                "Mention vector dimensionality "
                f"{len(vector)} does not match alias vector dimensionality "
                f"{self._dimensions}"
            )
        # Best score and its candidates per namespace, keeping only candidates at
        # or above the acceptance threshold.
        by_namespace: dict[str, list[tuple[float, str]]] = {}
        for alias in self._aliases:
            score = _similarity(vector, alias.vector, self._metric)
            if score >= self._threshold:
                by_namespace.setdefault(alias.namespace, []).append(
                    (score, alias.canonical_id)
                )
        return _resolve_scored_candidates(
            by_namespace, margin=self._margin, method=self._metric
        )


class VectorSimilarityResolver(EntityResolver):
    name = "vector_similarity"
    version = VECTOR_SIMILARITY_RESOLVER_VERSION

    def required_mention_columns(
        self, options: VectorSimilarityResolverOptions
    ) -> tuple[str, ...]:
        return (options.mention_vector_field,)

    def mention_signal(
        self, row: Mapping[str, Any], options: VectorSimilarityResolverOptions
    ) -> Any:
        return _coerce_vector(
            row[options.mention_vector_field],
            source="Mention",
            position=f"column '{options.mention_vector_field}'",
        )

    def validate_frames(
        self,
        mentions_frame: pl.DataFrame,
        aliases_frame: pl.DataFrame,
        options: VectorSimilarityResolverOptions,
    ) -> None:
        """Vectors from different embedding models/providers occupy unrelated
        coordinate spaces, so comparing them yields high-scoring but meaningless
        links. When both frames record an `embedding_config_hash`, require the
        spaces to match. Skipped when the column is absent (vectors from a source
        that does not record an embedding identity) or disabled with null."""
        hash_field = options.embedding_config_hash_field
        if hash_field is None:
            return
        mention_hashes = _distinct_values(mentions_frame, hash_field)
        alias_hashes = _distinct_values(aliases_frame, hash_field)
        if mention_hashes is None or alias_hashes is None:
            return
        spaces = mention_hashes | alias_hashes
        if len(spaces) > 1:
            raise ValueError(
                "vector_similarity requires the mention and alias embeddings to "
                f"share one embedding space, but column '{hash_field}' holds "
                f"{len(spaces)} distinct embedding-config identities. Embed both "
                "with the same provider/model/dimensions, or set "
                "embedding_config_hash_field: null to bypass this check."
            )

    def build_reference(
        self, frame: pl.DataFrame, options: VectorSimilarityResolverOptions
    ) -> _VectorReference:
        required = (
            options.alias_vector_field,
            options.namespace_field,
            options.canonical_id_field,
        )
        missing = sorted({field for field in required if field not in frame.columns})
        if missing:
            raise ValueError(
                f"Alias model '{options.aliases}' is missing configured columns "
                f"{missing}; got: {sorted(frame.columns)}"
            )
        aliases: list[_AliasVector] = []
        dimensions: int | None = None
        for position, row in enumerate(frame.iter_rows(named=True)):
            namespace = _reference_string(
                row, options.namespace_field, options.aliases, position
            )
            canonical_id = _reference_string(
                row, options.canonical_id_field, options.aliases, position
            )
            vector = _coerce_vector(
                row[options.alias_vector_field],
                source=f"Alias model '{options.aliases}'",
                position=f"row {position}",
            )
            if vector is None:
                raise ValueError(
                    f"Alias model '{options.aliases}' column "
                    f"'{options.alias_vector_field}' must contain a non-empty "
                    f"vector (row {position})"
                )
            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise ValueError(
                    f"Alias model '{options.aliases}' vectors must share one "
                    f"dimensionality; row {position} has {len(vector)}, expected "
                    f"{dimensions}"
                )
            aliases.append(_AliasVector(namespace, canonical_id, vector))
        return _VectorReference(aliases, dimensions or 0, options)


# --- Fuzzy resolver ----------------------------------------------------------


def _fuzzy_representation(text: str, metric: FuzzyMetric) -> frozenset[str]:
    """The metric's comparison signature for a prepared string. Character
    trigrams (space-padded so boundaries and short strings still produce grams)
    for ``trigram_dice``; whitespace tokens for ``jaccard_token``. An empty
    string yields an empty signature that matches nothing."""
    if not text:
        return frozenset()
    if metric == "jaccard_token":
        return frozenset(text.split())
    padded = f"  {text}  "
    return frozenset(padded[index : index + 3] for index in range(len(padded) - 2))


def _fuzzy_score(a: frozenset[str], b: frozenset[str], metric: FuzzyMetric) -> float:
    """Set-similarity in [0, 1]. Dice for trigrams, Jaccard for tokens; either
    empty signature scores 0 so empty text never matches."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if metric == "jaccard_token":
        return intersection / len(a | b)
    return 2 * intersection / (len(a) + len(b))


@dataclass(frozen=True)
class _FuzzyAlias:
    namespace: str
    canonical_id: str
    signature: frozenset[str]


class _FuzzyReference(ResolverReference):
    def __init__(
        self,
        aliases: list[_FuzzyAlias],
        options: FuzzyResolverOptions,
        fingerprint: str,
    ) -> None:
        self._aliases = aliases
        self._metric = options.metric
        self._normalize = options.normalize
        self._threshold = options.threshold
        self._margin = options.ambiguity_margin
        self._fingerprint = fingerprint

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _prepare(self, text: str) -> str:
        return normalize_alias_text(text) if self._normalize else text

    def resolve(self, signal: Any) -> dict[str, NamespaceResolution]:
        text = signal
        if not isinstance(text, str) or not text.strip():
            return {}
        mention_signature = _fuzzy_representation(self._prepare(text), self._metric)
        if not mention_signature:
            return {}
        by_namespace: dict[str, list[tuple[float, str]]] = {}
        for alias in self._aliases:
            score = _fuzzy_score(mention_signature, alias.signature, self._metric)
            if score >= self._threshold:
                by_namespace.setdefault(alias.namespace, []).append(
                    (score, alias.canonical_id)
                )
        return _resolve_scored_candidates(
            by_namespace, margin=self._margin, method=self._metric
        )


class FuzzyResolver(EntityResolver):
    name = "fuzzy"
    version = FUZZY_RESOLVER_VERSION

    def text_required(self) -> bool:
        return True

    def required_mention_columns(self, options: Any) -> tuple[str, ...]:
        return ()

    def mention_signal(self, row: Mapping[str, Any], options: Any) -> Any:
        raw = row[options.mention_text_field]
        if raw is not None and not isinstance(raw, str):
            raise ValueError(
                f"Mention text column '{options.mention_text_field}' must contain "
                "strings or nulls"
            )
        return raw

    def build_reference(
        self, frame: pl.DataFrame, options: FuzzyResolverOptions
    ) -> _FuzzyReference:
        alias_rows = _reference_rows(
            frame,
            options,
            value_fields=(
                ("alias", options.alias_text_field),
                ("entity_namespace", options.namespace_field),
                ("canonical_id", options.canonical_id_field),
            ),
        )
        aliases = [
            _FuzzyAlias(
                namespace=row["entity_namespace"],
                canonical_id=row["canonical_id"],
                signature=_fuzzy_representation(
                    normalize_alias_text(row["alias"])
                    if options.normalize
                    else row["alias"],
                    options.metric,
                ),
            )
            for row in alias_rows
        ]
        return _FuzzyReference(aliases, options, alias_set_fingerprint(alias_rows))


# --- Registry ----------------------------------------------------------------


RESOLVERS: dict[str, EntityResolver] = {
    resolver.name: resolver
    for resolver in (
        AliasTableResolver(),
        VectorSimilarityResolver(),
        FuzzyResolver(),
    )
}


def get_resolver(name: str) -> EntityResolver:
    try:
        return RESOLVERS[name]
    except KeyError as e:  # pragma: no cover - options validation rejects first
        raise ValueError(f"Unknown entity-linking resolver {name!r}") from e


# --- Shared helpers ----------------------------------------------------------


def _distinct_values(frame: pl.DataFrame, field: str) -> set[str] | None:
    """Distinct non-null string values of ``field``, or ``None`` when the column
    is absent or carries no non-null value (nothing to compare)."""
    if field not in frame.columns:
        return None
    values = {str(value) for value in frame[field].drop_nulls().to_list()}
    return values or None


def _reference_string(
    row: Mapping[str, Any], field: str, model_name: str, position: int
) -> str:
    raw = row[field]
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            f"Alias model '{model_name}' column '{field}' must contain "
            f"non-empty strings (row {position})"
        )
    return raw


def _reference_rows(
    frame: pl.DataFrame,
    options: EntityLinkBaseOptions,
    *,
    value_fields: tuple[tuple[str, str], ...],
) -> list[dict[str, str]]:
    fields = tuple(field for _, field in value_fields)
    missing = sorted({field for field in fields if field not in frame.columns})
    if missing:
        raise ValueError(
            f"Alias model '{options.aliases}' is missing configured columns "
            f"{missing}; got: {sorted(frame.columns)}"
        )
    rows: list[dict[str, str]] = []
    for position, row in enumerate(frame.iter_rows(named=True)):
        rows.append(
            {
                output_name: _reference_string(
                    row, field, options.aliases, position
                )
                for output_name, field in value_fields
            }
        )
    return rows


def entity_link_id(
    *,
    mention_id: str,
    document_id: str,
    entity_namespace: str | None,
    canonical_id: str | None,
    match_method: str | None,
    status: LinkStatus,
) -> str:
    return canonical_fingerprint(
        {
            "mention_id": mention_id,
            "document_id": document_id,
            "entity_namespace": entity_namespace or "",
            "canonical_id": canonical_id or "",
            "match_method": match_method or "",
            "status": status,
        },
        domain=ENTITY_LINK_FINGERPRINT_DOMAIN,
    )
