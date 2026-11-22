# ADR-0060: Language detection stays on langdetect

- **Status:** accepted
- **Date:** 2026-11-22
- **Prompted by:** [#802](https://github.com/C00ldudeNoonan/constellations/issues/802) (ALE-632, under the dependency-footprint theme ALE-625)

## Context

The `text` extra declares `langdetect` for one function,
`stel.text.language.detect_language`, reached through the
`stel.text.transforms.detect_language` transform. langdetect's last release
is 1.0.9, uploaded 2021-05-07, so it is unmaintained: an advisory or a Python
break would get no upstream fix. Its value is the bundled per-language n-gram
profiles, not its code, so a first-party detector would mean shipping and
owning that data. Auto language routing (ALE-309) will build on whichever
detector stel has.

## Decision

Keep `langdetect` behind the `text` extra and the one `detect_language`
function. Revisit only if it stops working on a supported Python, an advisory
lands against it, or ALE-309 needs a capability it lacks (per-script
confidence, or short-text accuracy). Because every caller goes through
`detect_language`, a swap later is a one-function change.

## Alternatives considered

### Replace with py3langid

A maintained fork of langid.py (0.4.0, 2026-09-02) with a bundled model. It
agreed with langdetect on 15 of 15 sample sentences, and returned `zh` where
langdetect returns `zh-cn`. It lost on footprint: the wheel is 4.6 MB against
1.0 MB, and it requires `numpy`, which is not in the `text` extra today, so a
text-only install would gain numpy to drop `six`. That moves the theme's goal
the wrong way. It also would change the output codes (`zh-cn` to `zh`) that
the documented transform emits.

### Replace with lingua-language-detector

Maintained and accurate on short text, but its wheels are about 170 MB
because the language models are compiled in. Far outside a lean optional
extra.

### Replace with fasttext-langdetect

The wrapper is small but it pulls `requests` and `numpy` and downloads a model
at first use, which breaks the offline and no-surprise-network posture of the
text extra.

### Drop the transform

Nothing in the repository's other code depends on it, and a search of the
astrolabe project finds no `detect_language` model or `langdetect` use. But
the transform is documented in `docs/reference.md` and pinned by tests, and
ALE-309 wants a detector, so removing it would be re-added shortly.

### Own the profiles first-party

Shipping and regenerating n-gram data for ~55 languages is a maintenance
burden larger than the problem it solves.

## Consequences

- `six` stays in `uv.lock` through langdetect.
- An unmaintained dependency stays in the `text` extra; the revisit triggers
  above are the guard. `pip-audit` runs in the standard workflow and would
  surface an advisory.
- `detect_language` keeps setting `DetectorFactory.seed = 0` per call so
  results stay deterministic.

## Evidence

- PyPI release history, read 2026-11-22: langdetect 1.0.9 (2021-05-07) is the
  newest of 1.0.7 (2016), 1.0.8 (2020) and 1.0.9; it requires only `six`.
- langdetect 1.0.9 imports and detects on Python 3.14 (run 2026-11-22), so
  there is no current Python break.
- Sample comparison (15 one-sentence inputs across en, de, fr, es, it, pt, nl,
  ja, zh, ru, ar, ko, sv, pl, tr): langdetect and py3langid 0.4.0 agree on
  15 of 15 at two-letter prefix. Wheel sizes from PyPI metadata: py3langid
  4.6 MB, lingua-language-detector about 170 MB, langdetect 1.0 MB.
- Astrolabe search for `detect_language` and `langdetect` in
  `*.py`, `*.yml`, `*.toml`, `*.sql`: no matches.
