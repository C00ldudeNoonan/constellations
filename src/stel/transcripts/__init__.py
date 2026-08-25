"""Agent transcript conversion: harness sessions → transcript/v1 landing files
(issue #360). See `contract` for the landing shape and `convert` for the
pipeline entry points."""
from .contract import (
    TRANSCRIPT_SCHEMA_VERSION,
    TranscriptContextCall,
    TranscriptDocument,
    TranscriptExchange,
)
from .convert import (
    DEFAULT_MIN_IDLE_SECONDS,
    build_document,
    convert_file,
    default_claude_dir,
    default_codex_dir,
    detect_harness,
    parse_transcript,
    sync_transcripts,
    write_landing_document,
)

__all__ = [
    "DEFAULT_MIN_IDLE_SECONDS",
    "TRANSCRIPT_SCHEMA_VERSION",
    "TranscriptContextCall",
    "TranscriptDocument",
    "TranscriptExchange",
    "build_document",
    "convert_file",
    "default_claude_dir",
    "default_codex_dir",
    "detect_harness",
    "parse_transcript",
    "sync_transcripts",
    "write_landing_document",
]
