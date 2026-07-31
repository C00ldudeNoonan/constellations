"""Entity-linking resolver contracts, registry, and the built-in resolvers.

Two resolvers ship today, both deterministic and offline:

- ``alias_table`` joins mention text to an operator-owned alias dimension with
  exact and normalized text matching. It never guesses: every mention outcome is
  an explicit ``matched``, ``ambiguous``, or ``unmatched`` status, and ambiguous
  candidates are preserved as separate rows.
- ``vector_similarity`` matches precomputed mention embeddings against
  precomputed alias embeddings by cosine/dot/euclidean similarity above a
  threshold. Both vectors are produced upstream by the ``embed`` model kind, so
  credentials, provider batching, and versioned embedding identity stay in that
  executor and this resolver remains a pure offline transform over frame data.
"""
from __future__ import annotations

import math
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import polars as pl
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    TypeAdapter,
    field_validator,
    model_validator,
)

from ..hashing import canonical_fingerprint

# Bumped whenever a resolver's matching semantics change (normalization rules,
# method precedence, status assignment, similarity math) so downstream consumers
# can invalidate rows produced by an older resolver even though the package
# version moved for unrelated reasons.
ALIAS_RESOLVER_VERSION = "1"
VECTOR_SIMILARITY_RESOLVER_VERSION = "1"
FUZZY_RESOLVER_VERSION = "1"

ALIAS_SET_FINGERPRINT_DOMAIN = "dbt-ml.entity-alias-set"
VECTOR_REFERENCE_FINGERPRINT_DOMAIN = "dbt-ml.entity-vector-reference-set"
ENTITY_LINK_FINGERPRINT_DOMAIN = "dbt-ml.entity-link"

MatchMethod = Literal["exact", "normalized"]
SimilarityMetric = Literal["cosine", "euclidean", "dot"]
FuzzyMetric = Literal["trigram_dice", "jaccard_token"]
LinkStatus = Literal["matched", "ambiguous", "unmatched"]


def _require_non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must not be empty")
    return normalized


class _EntityLinkBaseOptions(BaseModel):
    """Fields shared by every resolver: mention identity, privacy controls, and
    the passthrough/include projection. Resolver-specific matching inputs live on
    the concrete subclasses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mentions: str
    aliases: str
    mention_id_field: str = "entity_id"
    document_id_field: str = "document_id"
    mention_text_field: str = "entity_text"
    label_field: str | None = "label"
    start_field: str | None = "start"
    end_field: str | None = "end"
    namespace_field: str = "entity_namespace"
    canonical_id_field: str = "canonical_id"
    on_ambiguity: Literal["keep", "error"] = "keep"
    include_fields: tuple[str, ...] = ()
    include_mention_text: StrictBool = False

    @field_validator(
        "mentions",
        "aliases",
        "mention_id_field",
        "document_id_field",
        "mention_text_field",
        "namespace_field",
        "canonical_id_field",
    )
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("label_field", "start_field", "end_field")
    @classmethod
    def _non_empty_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty; use null to disable")
        return normalized

    @field_validator("include_fields")
    @classmethod
    def _unique_include_fields(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("include_fields entries must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("include_fields entries must be unique")
        return normalized

    @model_validator(mode="after")
    def _consistent_references(self) -> _EntityLinkBaseOptions:
        if self.mentions == self.aliases:
            raise ValueError(
                "mentions and aliases must reference two different upstream models"
            )
        forbidden = {
            field
            for field in (
                self.mention_id_field,
                self.document_id_field,
                self.mention_text_field,
            )
            if field in self.include_fields
        }
        if forbidden:
            raise ValueError(
                "include_fields must not repeat the mention ID, document ID, or "
                f"mention text field: {sorted(forbidden)}"
            )
        return self


class AliasTableResolverOptions(_EntityLinkBaseOptions):
    resolver: Literal["alias_table"] = "alias_table"
    alias_text_field: str = "alias"
    match_methods: tuple[MatchMethod, ...] = ("exact", "normalized")

    @field_validator("alias_text_field")
    @classmethod
    def _non_empty_alias_field(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("match_methods")
    @classmethod
    def _ordered_unique_methods(
        cls, values: tuple[MatchMethod, ...]
    ) -> tuple[MatchMethod, ...]:
        if not values:
            raise ValueError("match_methods must not be empty")
        if len(values) != len(set(values)):
            raise ValueError("match_methods entries must be unique")
        return values


class VectorSimilarityResolverOptions(_EntityLinkBaseOptions):
    resolver: Literal["vector_similarity"] = "vector_similarity"
    mention_vector_field: str = "embedding"
    alias_vector_field: str = "embedding"
    metric: SimilarityMetric = "cosine"
    # No default: matching by similarity is meaningless without an operator-chosen
    # acceptance bar, so require it explicitly rather than guessing one.
    threshold: float
    # Candidates within this margin of a namespace's top score are ambiguous
    # rather than silently resolved to the arg-max. 0.0 flags only exact ties.
    ambiguity_margin: float = 0.0
    # Guards against comparing vectors from unrelated embedding spaces: when both
    # sides carry this column (the `embed` kind emits `embedding_config_hash`), a
    # mismatch fails the run. Set to null to skip the check for vectors from a
    # source that does not record an embedding identity.
    embedding_config_hash_field: str | None = "embedding_config_hash"

    @field_validator("mention_vector_field", "alias_vector_field")
    @classmethod
    def _non_empty_vector_field(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("embedding_config_hash_field")
    @classmethod
    def _non_empty_optional_hash_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty; use null to disable")
        return normalized

    @field_validator("threshold")
    @classmethod
    def _finite_threshold(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("threshold must be a finite number")
        return value

    @field_validator("ambiguity_margin")
    @classmethod
    def _non_negative_margin(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("ambiguity_margin must be a finite, non-negative number")
        return value


class FuzzyResolverOptions(_EntityLinkBaseOptions):
    resolver: Literal["fuzzy"] = "fuzzy"
    alias_text_field: str = "alias"
    # `trigram_dice` (character-trigram Dice) is robust to spelling variants,
    # legal suffixes, and typos — the reason to reach past `alias_table`'s
    # exact/normalized matching. `jaccard_token` compares whitespace tokens and
    # suits reordered multi-word names. Both are in [0, 1], higher is better.
    metric: FuzzyMetric = "trigram_dice"
    # NFKC-fold, casefold, and whitespace-collapse both sides before scoring, so
    # fuzzy matching is case- and width-insensitive by default. Disable to score
    # the raw surface forms.
    normalize: StrictBool = True
    # No default: a similarity bar is meaningless without an operator-chosen
    # acceptance threshold, so require it explicitly.
    threshold: float
    # Candidates within this margin of a namespace's top score are ambiguous
    # rather than silently resolved to the arg-max. 0.0 flags only exact ties.
    ambiguity_margin: float = 0.0

    @field_validator("alias_text_field")
    @classmethod
    def _non_empty_alias_field(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("threshold")
    @classmethod
    def _threshold_in_unit_interval(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(
                "threshold must be a finite number in (0, 1]; fuzzy similarity is "
                "always in [0, 1]"
            )
        return value

    @field_validator("ambiguity_margin")
    @classmethod
    def _non_negative_margin(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("ambiguity_margin must be a finite, non-negative number")
        return value


EntityLinkConfig = Annotated[
    AliasTableResolverOptions
    | VectorSimilarityResolverOptions
    | FuzzyResolverOptions,
    Field(discriminator="resolver"),
]

# Back-compat alias: the pre-registry public name resolves to the default
# resolver's options model.
EntityLinkOptions = AliasTableResolverOptions

_ENTITY_LINK_ADAPTER: TypeAdapter[
    AliasTableResolverOptions | VectorSimilarityResolverOptions | FuzzyResolverOptions
] = TypeAdapter(EntityLinkConfig)


def parse_entity_link_options(
    options: Mapping[str, Any],
) -> AliasTableResolverOptions | VectorSimilarityResolverOptions | FuzzyResolverOptions:
    """Validate raw options into the resolver-specific model. ``resolver`` is
    optional and defaults to ``alias_table`` (phase-1 behavior); an unknown value
    fails discriminator validation with the valid tags named."""
    if isinstance(options, Mapping) and "resolver" not in options:
        options = {**options, "resolver": "alias_table"}
    return _ENTITY_LINK_ADAPTER.validate_python(options)


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
    options: _EntityLinkBaseOptions,
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
