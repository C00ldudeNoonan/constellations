"""Standard text-preprocessing primitives.

Importable directly:

    from dbt_ml.text import count_tokens, detect_language, text_stats

Or referenced from YAML as built-in transforms:

    transform:
      type: python
      module: dbt_ml.text.transforms.text_stats
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
from .language import detect_language
from .linking import (
    ALIAS_RESOLVER_VERSION,
    EntityLinkOptions,
    alias_set_fingerprint,
    entity_link_id,
    normalize_alias_text,
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
from .stats import text_stats
from .tokens import count_tokens

__all__ = [
    "ALIAS_RESOLVER_VERSION",
    "BASE_FEATURES",
    "DocumentFeatureOptions",
    "EntityLinkOptions",
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
    "alias_set_fingerprint",
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
    "pos_count_column",
    "pos_ratio_column",
    "redact_pii",
    "text_stats",
]
