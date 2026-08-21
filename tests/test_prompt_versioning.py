"""Named, versioned prompt artifacts (issue #303).

A prompt is a program input that changes the output, so it gets what code
gets: a name, a version, and a diff in review. These tests pin the resolution
rules (explicit version, compile-time failure, no traversal), the stamping
that lets a row say which prompt produced it, and the artifact rule that the
text never leaves the project.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from stel.config.model import FieldConfig, LLMTransformConfig, PromptRef
from stel.prompts import PromptError, ResolvedPrompt, resolve_prompt

# ─── the reference ──────────────────────────────────────────────────────────


def test_inline_prompts_still_work() -> None:
    # An additional form, not a replacement: inline is right for quick
    # projects and examples.
    config = LLMTransformConfig(input_field="text", prompt="classify this")

    resolved = resolve_prompt(config, Path("/nonexistent"), model_name="m")

    assert resolved.text == "classify this"
    assert resolved.name is None and resolved.version is None


def test_an_empty_inline_prompt_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        LLMTransformConfig(input_field="text", prompt="   ")


def test_a_reference_names_a_versioned_file(tmp_path: Path) -> None:
    _write_prompt(tmp_path, "signal_classify", "v3", "Classify the signal.")
    config = _ref_config("signal_classify", "v3")

    resolved = resolve_prompt(config, tmp_path, model_name="m")

    assert resolved.text == "Classify the signal."
    assert resolved.name == "signal_classify"
    assert resolved.version == "v3"


def test_path_segments_are_charset_validated() -> None:
    """Name and version become path segments, so traversal fails at config load.

    Sanitizing at read time would be later and weaker: the reviewed artifact
    is the YAML, and a `../` there should never parse.
    """
    for name, version in (
        ("../escape", "v1"),
        ("ok", "../v1"),
        ("ok", "v 1"),
        ("/abs", "v1"),
    ):
        with pytest.raises(ValueError, match="invalid"):
            PromptRef(name=name, version=version)


def test_there_is_no_latest_pointer() -> None:
    # A moving reference would make two runs of the same committed project
    # resolve to different text — the mutable-prompt problem versions fix.
    # `latest` is only ever a literal filename, never a resolution rule.
    assert "latest" not in PromptRef.model_fields


# ─── failures worth catching early ──────────────────────────────────────────


def test_a_missing_version_names_what_does_exist(tmp_path: Path) -> None:
    _write_prompt(tmp_path, "signal_classify", "v1", "one")
    _write_prompt(tmp_path, "signal_classify", "v2", "two")
    config = _ref_config("signal_classify", "v9")

    with pytest.raises(PromptError) as excinfo:
        resolve_prompt(config, tmp_path, model_name="classified")

    message = str(excinfo.value)
    assert "signal_classify/v9" in message
    assert "v1, v2" in message


def test_an_unknown_name_says_so(tmp_path: Path) -> None:
    config = _ref_config("never_written", "v1")

    with pytest.raises(PromptError, match="No versions of 'never_written'"):
        resolve_prompt(config, tmp_path, model_name="m")


def test_an_empty_version_file_is_a_typo_not_a_prompt(tmp_path: Path) -> None:
    _write_prompt(tmp_path, "blank", "v1", "   \n  ")
    config = _ref_config("blank", "v1")

    with pytest.raises(PromptError, match="is empty"):
        resolve_prompt(config, tmp_path, model_name="m")


@pytest.mark.skipif(
    not hasattr(Path, "symlink_to"), reason="platform without symlinks"
)
def test_a_symlinked_prompt_is_refused(tmp_path: Path) -> None:
    """Same rule as project configuration: regular non-symlink files only.

    Caught by the path-component walk rather than the later stat check — the
    walk covers every component, so it reaches a symlinked leaf first. The
    stat check still guards non-regular files (a fifo, say).
    """
    outside = tmp_path / "outside.md"
    outside.write_text("text from outside the project")
    directory = tmp_path / "project" / "prompts" / "sneaky"
    directory.mkdir(parents=True)
    try:
        (directory / "v1.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation not permitted")

    with pytest.raises(PromptError, match="is a symlink"):
        resolve_prompt(
            _ref_config("sneaky", "v1"), tmp_path / "project", model_name="m"
        )


# ─── identity is artifact-safe ──────────────────────────────────────────────


def test_identity_carries_name_and_version_never_text() -> None:
    resolved = ResolvedPrompt(text="SECRET", name="n", version="v1")

    assert resolved.identity() == {"prompt_name": "n", "prompt_version": "v1"}
    assert "SECRET" not in json.dumps(resolved.identity())


# ─── end to end ─────────────────────────────────────────────────────────────

CANARY = "CANARY_PROMPT_TEXT_XYZ"


def _write_prompt(project: Path, name: str, version: str, text: str) -> None:
    directory = project / "prompts" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{version}.md").write_text(text, encoding="utf-8")


def _ref_config(name: str, version: str) -> LLMTransformConfig:
    return LLMTransformConfig(
        input_field="body", prompt=PromptRef(name=name, version=version)
    )


def _project(tmp_path: Path, *, prompt_yaml: str, run_log: bool = False) -> Path:
    project = tmp_path / "proj"
    (project / "models").mkdir(parents=True)
    (project / "sources").mkdir()
    docs = project / "data" / "docs"
    docs.mkdir(parents=True)

    (project / "stel_project.yml").write_text(
        "name: pp\nversion: '0.1.0'\nprofile: pp\n"
    )
    log_block = "      run_log:\n        enabled: true\n" if run_log else ""
    (project / "profiles.yml").write_text(
        "pp:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n"
        "        schema: main\n"
        "      llm:\n        provider: deterministic\n"
        "        model: deterministic-v1\n" + log_block
    )
    (project / "sources" / "src.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data/docs\n"
        "    file_pattern: '*.json'\n"
    )
    (docs / "a.json").write_text(json.dumps({"note_id": "n1", "body": "hello"}))
    (project / "models" / "m.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: notes\n    source: ref('docs')\n"
        "    extraction:\n      backend: json\n      options:\n"
        "        fields: [note_id, body]\n"
        "  - name: classified\n    depends_on: [ref('notes')]\n"
        "    llm:\n      mode: map\n      input_field: body\n"
        "      id_field: note_id\n"
        f"      prompt: {prompt_yaml}\n"
        "    fields:\n      - name: signal\n        type: string\n"
    )
    return project


def test_rows_record_which_prompt_produced_them(tmp_path: Path) -> None:
    """`llm_config_hash` says something changed; this says what ran."""
    from stel.runner import run_project

    project = _project(tmp_path, prompt_yaml="{ name: signal_classify, version: v3 }")
    _write_prompt(project, "signal_classify", "v3", "Classify the note.")

    run_project(project)

    con = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        row = con.execute(
            "SELECT prompt_name, prompt_version FROM main.classified"
        ).fetchone()
    finally:
        con.close()
    assert row == ("signal_classify", "v3")


def test_an_inline_prompt_leaves_the_columns_null(tmp_path: Path) -> None:
    # There is no stable identity to record — which is the gap versioned
    # prompts close, stated in the data rather than implied.
    from stel.runner import run_project

    project = _project(tmp_path, prompt_yaml="'classify this note'")

    run_project(project)

    con = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        row = con.execute(
            "SELECT prompt_name, prompt_version FROM main.classified"
        ).fetchone()
    finally:
        con.close()
    assert row == (None, None)


def test_compile_fails_on_a_misspelled_version(tmp_path: Path) -> None:
    """A typo costs nothing, rather than costing a corpus.

    Resolution has to be its own compile-time check: the artifact-safe
    descriptor path swallows failures so offline docs tooling still works, so
    a bad reference would otherwise survive compile and surface only after
    source discovery and credential resolution.
    """
    from stel.compiler import validate_project_contract
    from stel.config import ConfigError, load_project

    project = _project(tmp_path, prompt_yaml="{ name: signal_classify, version: v9 }")
    _write_prompt(project, "signal_classify", "v1", "Classify the note.")
    config, sources, models = load_project(project)

    with pytest.raises(ConfigError, match="signal_classify/v9"):
        validate_project_contract(config, sources, models, project)


def test_the_prompt_text_never_reaches_the_manifest(tmp_path: Path) -> None:
    """The documented rule, now actually true for native `llm:` models.

    An inline prompt reached manifest.json verbatim before this change —
    verified against the built artifact, not by reading the serializer.
    """
    from stel.manifest import write_manifest

    project = _project(tmp_path, prompt_yaml=f"'{CANARY} classify this'")

    write_manifest(project)

    manifest = (project / "target" / "manifest.json").read_text()
    assert CANARY not in manifest


def test_the_manifest_records_the_reference(tmp_path: Path) -> None:
    from stel.manifest import write_manifest

    project = _project(tmp_path, prompt_yaml="{ name: signal_classify, version: v3 }")
    _write_prompt(project, "signal_classify", "v3", f"{CANARY} classify")

    write_manifest(project)

    manifest = json.loads((project / "target" / "manifest.json").read_text())
    block = next(m for m in manifest["models"] if m["name"] == "classified")["llm"]
    assert block["prompt_name"] == "signal_classify"
    assert block["prompt_version"] == "v3"
    assert CANARY not in json.dumps(manifest)


def test_the_run_log_groups_cost_by_prompt_version(tmp_path: Path) -> None:
    """#306's headline query, answerable for the first time.

    The run log specified these columns; they stayed null until there were
    versioned prompts to fill them.
    """
    from stel.runner import run_project

    project = _project(
        tmp_path,
        prompt_yaml="{ name: signal_classify, version: v3 }",
        run_log=True,
    )
    _write_prompt(project, "signal_classify", "v3", "Classify the note.")

    run_project(project)

    con = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        rows = dict(
            con.execute(
                "SELECT model_name, prompt_version FROM main.stel_run_log"
            ).fetchall()
        )
    finally:
        con.close()
    assert rows["classified"] == "v3"
    # A model with no prompt at all records none, rather than an empty string.
    assert rows["notes"] is None


def test_editing_a_version_in_place_still_invalidates(tmp_path: Path) -> None:
    """Until the immutability gate lands, the hash must stay correct.

    Editing a released version is meant to become a failed build (deferred
    follow-up). In the meantime it must at least not produce stale rows.
    """
    from stel.config import load_project
    from stel.llm_map import resolve_llm_runtime
    from stel.profile import resolve_profile

    project = _project(tmp_path, prompt_yaml="{ name: signal_classify, version: v3 }")
    _write_prompt(project, "signal_classify", "v3", "First instruction.")
    config, _, models = load_project(project)
    resolved = resolve_profile(config, project)
    model = next(m for m in models if m.name == "classified")
    assert model.llm is not None

    before = resolve_llm_runtime(
        model.llm, model.fields, resolved, project_dir=project, model_name=model.name
    ).config_hash

    _write_prompt(project, "signal_classify", "v3", "Second, different instruction.")
    after = resolve_llm_runtime(
        model.llm, model.fields, resolved, project_dir=project, model_name=model.name
    ).config_hash

    assert before != after


# ─── the immutability gate (issue #303) ─────────────────────────────────────


def test_a_clean_tree_passes_the_gate(tmp_path: Path) -> None:
    from stel.prompts import check_lock, write_lock

    _write_prompt(tmp_path, "signal_classify", "v1", "First instruction.")
    write_lock(tmp_path)

    assert check_lock(tmp_path) == []


def test_editing_a_released_version_fails_the_gate(tmp_path: Path) -> None:
    """The point of the feature: a failed build, not a silent reprocess."""
    from stel.prompts import check_lock, write_lock

    _write_prompt(tmp_path, "signal_classify", "v1", "First instruction.")
    write_lock(tmp_path)

    _write_prompt(tmp_path, "signal_classify", "v1", "Edited in place.")
    drift = check_lock(tmp_path)

    assert [(d.name, d.version, d.kind) for d in drift] == [
        ("signal_classify", "v1", "changed")
    ]
    assert "Add the next version" in drift[0].describe()


def test_adding_the_next_version_is_the_frictionless_path(tmp_path: Path) -> None:
    from stel.prompts import check_lock, write_lock

    _write_prompt(tmp_path, "signal_classify", "v1", "First instruction.")
    write_lock(tmp_path)

    _write_prompt(tmp_path, "signal_classify", "v2", "Second instruction.")
    added, rewritten = write_lock(tmp_path)

    assert (added, rewritten) == (1, 0)
    assert check_lock(tmp_path) == []


def test_lock_refuses_to_launder_an_edit(tmp_path: Path) -> None:
    """`lock` must not be a one-command bypass of the gate.

    If re-locking silently accepted a changed release, the gate would teach
    exactly the workflow it exists to prevent.
    """
    from stel.prompts import PromptLockError, write_lock

    _write_prompt(tmp_path, "signal_classify", "v1", "First instruction.")
    write_lock(tmp_path)
    _write_prompt(tmp_path, "signal_classify", "v1", "Edited in place.")

    with pytest.raises(PromptLockError, match="add the next version instead"):
        write_lock(tmp_path)


def test_force_relocks_and_reports_what_it_did(tmp_path: Path) -> None:
    # Deliberate and reviewed: the lock diff is the review artifact.
    from stel.prompts import check_lock, write_lock

    _write_prompt(tmp_path, "signal_classify", "v1", "First instruction.")
    write_lock(tmp_path)
    _write_prompt(tmp_path, "signal_classify", "v1", "Edited in place.")

    added, rewritten = write_lock(tmp_path, force=True)

    assert (added, rewritten) == (0, 1)
    assert check_lock(tmp_path) == []


def test_an_unlocked_version_is_reported(tmp_path: Path) -> None:
    from stel.prompts import check_lock, write_lock

    _write_prompt(tmp_path, "signal_classify", "v1", "First.")
    write_lock(tmp_path)
    _write_prompt(tmp_path, "signal_classify", "v2", "Second.")

    drift = check_lock(tmp_path)

    assert [(d.version, d.kind) for d in drift] == [("v2", "unlocked")]
    assert "stel prompts lock" in drift[0].describe()


def test_deleting_a_released_version_is_reported(tmp_path: Path) -> None:
    """Rows already produced under it record that version."""
    from stel.prompts import check_lock, write_lock

    _write_prompt(tmp_path, "signal_classify", "v1", "First.")
    write_lock(tmp_path)
    (tmp_path / "prompts" / "signal_classify" / "v1.md").unlink()

    drift = check_lock(tmp_path)

    assert [(d.version, d.kind) for d in drift] == [("v1", "missing_file")]


def test_trailing_whitespace_is_not_an_edit(tmp_path: Path) -> None:
    # The instruction is what is protected, not the file's whitespace — an
    # editor adding a final newline must not read as a released-prompt edit.
    from stel.prompts import check_lock, write_lock

    _write_prompt(tmp_path, "signal_classify", "v1", "First instruction.")
    write_lock(tmp_path)

    _write_prompt(tmp_path, "signal_classify", "v1", "First instruction.\n\n")

    assert check_lock(tmp_path) == []


def test_a_hand_corrupted_lock_says_how_to_recover(tmp_path: Path) -> None:
    from stel.prompts import PromptLockError, lock_path, read_lock, write_lock

    _write_prompt(tmp_path, "signal_classify", "v1", "First.")
    write_lock(tmp_path)
    lock_path(tmp_path).write_text("{not json", encoding="utf-8")

    with pytest.raises(PromptLockError, match="stel prompts lock"):
        read_lock(tmp_path)


def test_no_prompts_directory_is_not_a_failure(tmp_path: Path) -> None:
    # A project with only inline prompts must not be told to run a lock.
    from stel.prompts import check_lock

    assert check_lock(tmp_path) == []


def test_the_cli_gate_exits_nonzero_on_a_changed_release(tmp_path: Path) -> None:
    """CI runs this; the exit code is the contract."""
    from click.testing import CliRunner

    from stel.cli import cli
    from stel.prompts import write_lock

    _write_prompt(tmp_path, "signal_classify", "v1", "First instruction.")
    write_lock(tmp_path)
    _write_prompt(tmp_path, "signal_classify", "v1", "Edited in place.")

    result = CliRunner().invoke(
        cli, ["prompts", "check", "--project-dir", str(tmp_path)]
    )

    assert result.exit_code != 0
    assert "was released and has since changed" in result.output

# ─── review follow-ups (PR #334) ────────────────────────────────────────────


def _skip_without_symlinks(target: Path, link: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError:  # pragma: no cover - Windows without privilege
        pytest.skip("symlink creation not permitted")


def test_a_symlinked_prompt_directory_is_refused(tmp_path: Path) -> None:
    """The leaf check is not enough.

    A symlinked `prompts/<name>/` leaves `<version>.md` an ordinary regular
    file, so checking only the final component passes while the read lands
    outside the project — and that text goes to an inference provider.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "v1.md").write_text("operator-local data")
    project = tmp_path / "project"
    (project / "prompts").mkdir(parents=True)
    _skip_without_symlinks(outside, project / "prompts" / "sneaky")

    with pytest.raises(PromptError, match="is a symlink"):
        resolve_prompt(_ref_config("sneaky", "v1"), project, model_name="m")


def test_an_inline_prompt_keeps_its_existing_config_hash() -> None:
    """Adding prompt identity must not re-key every existing `llm:` model.

    Two null keys in the hash payload would have reprocessed every corpus to
    record that nothing was named — the same class of churn the backend
    identity pin exists to prevent. Pinned to the value master produces,
    verified against a master worktree.
    """
    from stel.llm_map import resolve_llm_runtime

    config = LLMTransformConfig(
        input_field="t",
        prompt="classify",
        provider="deterministic",
        model="deterministic-v1",
    )
    runtime = resolve_llm_runtime(
        config, [FieldConfig(name="x", data_type="string")], None
    )

    assert runtime.config_hash == "1483ec56eb703c4d242c4686bbc0b0ae"


def test_a_versioned_prompt_does_change_the_hash(tmp_path: Path) -> None:
    from stel.llm_map import resolve_llm_runtime

    _write_prompt(tmp_path, "signal_classify", "v1", "classify")
    versioned = resolve_llm_runtime(
        _ref_config("signal_classify", "v1"),
        [FieldConfig(name="x", data_type="string")],
        None,
        project_dir=tmp_path,
        model_name="m",
    )

    assert versioned.config_hash != "1483ec56eb703c4d242c4686bbc0b0ae"


def test_an_append_only_log_widens_instead_of_breaking(tmp_path: Path) -> None:
    """A log created by an earlier release must survive a new column.

    Log writes are best-effort, so a rejected write is silent — the durable
    history would simply stop at the upgrade.
    """
    import polars as pl

    from stel.adapters import create_adapter, parse_warehouse_config

    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "l.duckdb"), "schema": "main"}
    )
    with create_adapter(config) as adapter:
        adapter.append_rows("stel_run_log", pl.DataFrame([{"model_name": "a"}]))

        adapter.append_rows(
            "stel_run_log",
            pl.DataFrame([{"model_name": "b", "prompt_version": "v3"}]),
        )

        rows = adapter.read_table("stel_run_log").sort("model_name").to_dicts()
    assert rows == [
        {"model_name": "a", "prompt_version": None},
        {"model_name": "b", "prompt_version": "v3"},
    ]


# ─── review follow-ups (PR #336) ────────────────────────────────────────────


def test_lock_refuses_to_launder_a_deletion(tmp_path: Path) -> None:
    """Deleting a release and locking something else must not erase it.

    Otherwise the missing-file protection is one unrelated `lock` away from
    gone, and `check` then reports success.
    """
    from stel.prompts import PromptLockError, write_lock

    _write_prompt(tmp_path, "signal_classify", "v1", "First.")
    write_lock(tmp_path)
    (tmp_path / "prompts" / "signal_classify" / "v1.md").unlink()
    _write_prompt(tmp_path, "signal_classify", "v2", "Second.")

    with pytest.raises(PromptLockError, match=r"file\(s\) gone"):
        write_lock(tmp_path)


def test_an_unrecognized_lock_format_fails_closed(tmp_path: Path) -> None:
    """A gate that passes on a schema it cannot read is worse than none."""
    from stel.prompts import PromptLockError, lock_path, read_lock, write_lock

    _write_prompt(tmp_path, "signal_classify", "v1", "First.")
    write_lock(tmp_path)
    payload = json.loads(lock_path(tmp_path).read_text())
    payload["version"] = 999
    lock_path(tmp_path).write_text(json.dumps(payload))

    with pytest.raises(PromptLockError, match="format version"):
        read_lock(tmp_path)


def test_a_malformed_prompts_mapping_fails_closed(tmp_path: Path) -> None:
    from stel.prompts import PromptLockError, lock_path, read_lock

    (tmp_path / "prompts").mkdir()
    lock_path(tmp_path).write_text(
        json.dumps({"version": 1, "prompts": {"a/v1": 5}})
    )

    with pytest.raises(PromptLockError, match="malformed"):
        read_lock(tmp_path)


def test_a_symlinked_lock_is_never_written_through(tmp_path: Path) -> None:
    """`write_text` follows a symlink and would truncate its target."""
    from stel.prompts import PromptLockError, lock_path, write_lock

    victim = tmp_path / "important.txt"
    victim.write_text("do not truncate me")
    _write_prompt(tmp_path, "signal_classify", "v1", "First.")
    _skip_without_symlinks(victim, lock_path(tmp_path))

    with pytest.raises(PromptLockError, match="symlink"):
        write_lock(tmp_path)

    assert victim.read_text() == "do not truncate me"


def test_a_lock_path_failure_surfaces_as_a_lock_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI catches `PromptLockError`, so a path failure must be one.

    Otherwise a symlinked lock reaches the user as an unhandled traceback.
    Written without a real symlink deliberately: the symlink tests skip on
    Windows, so this conversion would otherwise be verified only in CI.
    """
    from stel import prompts as prompts_module
    from stel.prompts import PromptError, PromptLockError, read_lock

    def _refuse(*args: object, **kwargs: object) -> Path:
        raise PromptError("Refusing to use the prompt lock: 'x' is a symlink.")

    monkeypatch.setattr(prompts_module, "verify_project_path", _refuse)

    with pytest.raises(PromptLockError, match="is a symlink"):
        read_lock(tmp_path)


def test_the_cli_reports_a_lock_path_failure_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    from stel import prompts as prompts_module
    from stel.cli import cli
    from stel.prompts import PromptError

    def _refuse(*args: object, **kwargs: object) -> Path:
        raise PromptError("Refusing to use the prompt lock: 'x' is a symlink.")

    monkeypatch.setattr(prompts_module, "verify_project_path", _refuse)

    result = CliRunner().invoke(
        cli, ["prompts", "check", "--project-dir", str(tmp_path)]
    )

    assert result.exit_code != 0
    assert "is a symlink" in result.output
    # A clean error, not a traceback.
    assert result.exception is None or isinstance(result.exception, SystemExit)
