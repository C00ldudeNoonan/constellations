from __future__ import annotations

from ..optional_dependencies import import_optional_dependency


def detect_language(text: str, *, default: str | None = None) -> str | None:
    """Return a 2-letter ISO 639-1 language code for `text`, or `default` if
    detection fails or input is too short.

    Uses langdetect (Naive Bayes; ~55 supported languages). Short inputs
    (<10 non-whitespace chars) return `default` without invoking detection
    because the detector hallucinates wildly on tiny strings.
    """
    if not text or len(text.strip()) < 10:
        return default
    langdetect = import_optional_dependency(
        "langdetect", extra="text", feature="Language detection"
    )

    try:
        # Make detection deterministic across calls.
        langdetect.DetectorFactory.seed = 0
        return str(langdetect.detect(text))
    except Exception:
        return default
