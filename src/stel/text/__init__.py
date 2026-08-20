"""Standard text-preprocessing primitives.

Importable directly:

    from stel.text import count_tokens, detect_language, text_stats

Or referenced from YAML as built-in transforms:

    transform:
      type: python
      module: stel.text.transforms.text_stats
      options:
        text_field: body
"""
from .dedup import minhash_signature, near_duplicates
from .encoding import clean_encoding
from .features import (
    BASE_FEATURES,
    DocumentFeatureOptions,
    entity_label_column,
    link_namespace_column,
    link_status_column,
    pos_count_column,
    pos_ratio_column,
)
from .keyphrases import (
    KEYPHRASE_DOMAIN,
    KEYPHRASE_EXTRACTOR_NAME,
    KEYPHRASE_EXTRACTOR_VERSION,
    KeyphraseOptions,
)
from .language import detect_language
from .linking import (
    ALIAS_RESOLVER_VERSION,
    FUZZY_RESOLVER_VERSION,
    VECTOR_SIMILARITY_RESOLVER_VERSION,
    AliasTableResolverOptions,
    EntityLinkConfig,
    EntityLinkOptions,
    FuzzyResolverOptions,
    VectorSimilarityResolverOptions,
    alias_set_fingerprint,
    entity_link_id,
    normalize_alias_text,
    parse_entity_link_options,
)
from .nlp import (
    NLPDocument,
    NLPEntity,
    NLPEntityOptions,
    NLPError,
    NLPIdentity,
    NLPProvider,
    NLPToken,
    NLPTokenOptions,
    get_nlp_provider,
)
from .pii import PIIEntity, PIIError, detect_pii, redact_pii
from .relations import (
    CO_OCCURRENCE_EXTRACTOR_VERSION,
    MODEL_ASSERTION_EXTRACTOR_VERSION,
    RULE_EXTRACTOR_VERSION,
    CoOccurrenceExtractorOptions,
    ModelAssertionExtractorOptions,
    RelationExtractorConfig,
    RelationRule,
    RuleExtractorOptions,
    parse_relation_options,
    relation_id,
)
from .stats import text_stats
from .tokens import count_tokens
from .tone import (
    TONE_SCORER_VERSION,
    ToneOptions,
    category_hits_column,
    category_score_column,
    tone_lexicon_fingerprint,
)

__all__ = [
    "ALIAS_RESOLVER_VERSION",
    "BASE_FEATURES",
    "CO_OCCURRENCE_EXTRACTOR_VERSION",
    "FUZZY_RESOLVER_VERSION",
    "KEYPHRASE_DOMAIN",
    "KEYPHRASE_EXTRACTOR_NAME",
    "KEYPHRASE_EXTRACTOR_VERSION",
    "MODEL_ASSERTION_EXTRACTOR_VERSION",
    "RULE_EXTRACTOR_VERSION",
    "TONE_SCORER_VERSION",
    "VECTOR_SIMILARITY_RESOLVER_VERSION",
    "AliasTableResolverOptions",
    "CoOccurrenceExtractorOptions",
    "DocumentFeatureOptions",
    "EntityLinkConfig",
    "EntityLinkOptions",
    "FuzzyResolverOptions",
    "KeyphraseOptions",
    "ModelAssertionExtractorOptions",
    "NLPDocument",
    "NLPEntity",
    "NLPEntityOptions",
    "NLPError",
    "NLPIdentity",
    "NLPProvider",
    "NLPToken",
    "NLPTokenOptions",
    "PIIEntity",
    "PIIError",
    "RelationExtractorConfig",
    "RelationRule",
    "RuleExtractorOptions",
    "ToneOptions",
    "VectorSimilarityResolverOptions",
    "alias_set_fingerprint",
    "category_hits_column",
    "category_score_column",
    "clean_encoding",
    "count_tokens",
    "detect_language",
    "detect_pii",
    "entity_label_column",
    "entity_link_id",
    "get_nlp_provider",
    "link_namespace_column",
    "link_status_column",
    "minhash_signature",
    "near_duplicates",
    "normalize_alias_text",
    "parse_entity_link_options",
    "parse_relation_options",
    "pos_count_column",
    "pos_ratio_column",
    "redact_pii",
    "relation_id",
    "text_stats",
    "tone_lexicon_fingerprint",
]
