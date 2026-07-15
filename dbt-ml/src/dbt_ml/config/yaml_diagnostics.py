from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from yaml.nodes import MappingNode, Node, SequenceNode

type ConfigPath = tuple[str | int, ...]


@dataclass(frozen=True)
class SourcePosition:
    line: int
    column: int


@dataclass(frozen=True)
class _PathPositions:
    key: SourcePosition
    value: SourcePosition


@dataclass(frozen=True)
class YamlDocument:
    data: Any
    _positions: dict[ConfigPath, _PathPositions]

    def without_data(self) -> YamlDocument:
        return YamlDocument(data=None, _positions=self._positions)

    def format_validation_errors(
        self,
        path: Path,
        error: ValidationError,
        *,
        prefix: ConfigPath = (),
    ) -> str:
        return self.format_validation_details(
            path,
            error.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            ),
            prefix=prefix,
        )

    def format_validation_details(
        self,
        path: Path,
        details: Iterable[Mapping[str, Any]],
        *,
        prefix: ConfigPath = (),
    ) -> str:
        diagnostics: list[str] = []
        for detail in details:
            detail_location = tuple(
                part if isinstance(part, str | int) else "<unknown>"
                for part in detail.get("loc", ())
            )
            location = prefix + detail_location
            position = self._position_for(location, str(detail["type"]))
            config_path = _render_config_path(location)
            diagnostics.append(
                f"  {path}:{position.line}:{position.column} "
                f"[{config_path}] {detail['msg']} [type={detail['type']}]"
            )
        return "\n".join(diagnostics)

    def format_message(
        self,
        path: Path,
        config_path: ConfigPath,
        message: str,
    ) -> str:
        position = self._position_for(config_path, "value_error")
        rendered_path = _render_config_path(config_path)
        return (
            f"{path}:{position.line}:{position.column} "
            f"[{rendered_path}] {message}"
        )

    def _position_for(
        self, config_path: ConfigPath, error_type: str
    ) -> SourcePosition:
        exact = self._positions.get(config_path)
        if exact is not None:
            if error_type in {"extra_forbidden", "invalid_key"}:
                return exact.key
            return exact.value

        for length in range(len(config_path) - 1, -1, -1):
            ancestor = self._positions.get(config_path[:length])
            if ancestor is not None:
                return ancestor.value
        return SourcePosition(line=1, column=1)


@dataclass(frozen=True)
class YamlProvenance:
    file_path: Path
    config_path: ConfigPath
    _document: YamlDocument = field(repr=False, compare=False)

    def format_message(
        self,
        message: str,
        *,
        relative_path: ConfigPath = (),
    ) -> str:
        return self._document.format_message(
            self.file_path,
            (*self.config_path, *relative_path),
            message,
        )


class DuplicateKeyError(yaml.MarkedYAMLError):
    def __init__(self, key_mark: Any, config_path: ConfigPath) -> None:
        super().__init__(
            problem="duplicate mapping key",
            problem_mark=key_mark,
        )
        self.config_path = config_path


def parse_yaml_document(text: str) -> YamlDocument:
    loader = yaml.SafeLoader(text)
    try:
        root = loader.get_single_node()
        if root is None:
            return YamlDocument(data=None, _positions={})
        _reject_duplicate_keys(loader, root, (), frozenset())
        data = loader.construct_document(root)  # type: ignore[no-untyped-call]
        positions: dict[ConfigPath, _PathPositions] = {}
        root_position = _source_position(root)
        positions[()] = _PathPositions(key=root_position, value=root_position)
        _collect_positions(loader, root, (), positions, frozenset())
        return YamlDocument(data=data, _positions=positions)
    finally:
        loader.dispose()  # type: ignore[no-untyped-call]


def format_yaml_parse_error(path: Path, error: yaml.YAMLError) -> str:
    mark = getattr(error, "problem_mark", None)
    problem = getattr(error, "problem", None) or type(error).__name__
    kind = "Invalid YAML" if isinstance(error, DuplicateKeyError) else "Malformed YAML"
    config_path = getattr(error, "config_path", None)
    path_suffix = (
        f" [{_render_config_path(config_path)}]" if config_path is not None else ""
    )
    if mark is None:
        return f"{kind} at {path}{path_suffix}: {problem}"
    return (
        f"{kind} at {path}:{mark.line + 1}:{mark.column + 1}"
        f"{path_suffix}: {problem}"
    )


def _reject_duplicate_keys(
    loader: yaml.SafeLoader,
    node: Node,
    path: ConfigPath,
    ancestors: frozenset[int],
) -> None:
    node_id = id(node)
    if node_id in ancestors:
        return
    nested_ancestors = ancestors | {node_id}

    if isinstance(node, MappingNode):
        seen: set[Any] = set()
        merge_seen = False
        for key_node, value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                if merge_seen:
                    raise DuplicateKeyError(key_node.start_mark, (*path, "<<"))
                merge_seen = True
                _reject_duplicate_keys(
                    loader,
                    value_node,
                    path,
                    nested_ancestors,
                )
                continue
            key = loader.construct_object(key_node, deep=True)
            try:
                duplicate = key in seen
            except TypeError:
                duplicate = False
            path_part = key if isinstance(key, str | int) else str(key)
            child_path = (*path, path_part)
            if duplicate:
                raise DuplicateKeyError(key_node.start_mark, child_path)
            try:
                seen.add(key)
            except TypeError:
                pass
            _reject_duplicate_keys(
                loader,
                value_node,
                child_path,
                nested_ancestors,
            )
        return

    if isinstance(node, SequenceNode):
        for index, value_node in enumerate(node.value):
            _reject_duplicate_keys(
                loader,
                value_node,
                (*path, index),
                nested_ancestors,
            )


def _collect_positions(
    loader: yaml.SafeLoader,
    node: Node,
    path: ConfigPath,
    positions: dict[ConfigPath, _PathPositions],
    ancestors: frozenset[int],
) -> None:
    node_id = id(node)
    if node_id in ancestors:
        return
    nested_ancestors = ancestors | {node_id}

    if isinstance(node, MappingNode):
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            if not isinstance(key, str | int):
                continue
            child_path = (*path, key)
            positions[child_path] = _PathPositions(
                key=_source_position(key_node),
                value=_source_position(value_node),
            )
            _collect_positions(
                loader,
                value_node,
                child_path,
                positions,
                nested_ancestors,
            )
        return

    if isinstance(node, SequenceNode):
        for index, value_node in enumerate(node.value):
            child_path = (*path, index)
            position = _source_position(value_node)
            positions[child_path] = _PathPositions(key=position, value=position)
            _collect_positions(
                loader,
                value_node,
                child_path,
                positions,
                nested_ancestors,
            )


def _source_position(node: Node) -> SourcePosition:
    return SourcePosition(
        line=node.start_mark.line + 1,
        column=node.start_mark.column + 1,
    )


def _render_config_path(config_path: ConfigPath) -> str:
    return ".".join(str(part) for part in config_path) or "<root>"
