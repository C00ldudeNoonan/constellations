"""Resolve `ontologies:` into the project's declaration at load (issue #767).

Each `ontologies:` entry names a Turtle or RDF/XML file inside the project and
an allow-list of IRIs. This module parses the file and turns each allow-listed
IRI into the declaration a project could have written by hand:

- a `skos:Concept` or `owl:NamedIndividual` becomes a vocabulary term:
  `skos:prefLabel` (else `rdfs:label`) is its label, `skos:altLabel` its
  aliases, `skos:definition` its description, `skos:broader` (or an inverse
  `skos:narrower`) its broader term, and an `rdf:type` naming an allow-listed
  class its `class:`;
- an `owl:Class` becomes a class, with an allow-listed `rdfs:subClassOf` as
  its `subclass_of`;
- an `owl:ObjectProperty` becomes one relation pairing per allow-listed
  `rdfs:domain` x `rdfs:range` class (an `owl:unionOf` domain or range counts
  each named member), `functional` when it is also an `owl:FunctionalProperty`.

Each keeps its source IRI. Hierarchy is the allow-list's induced subgraph: a
`broader` or `subClassOf` link to an IRI outside the allow-list is not
followed, which is what an allow-list means. An allow-listed IRI the file does
not declare fails here, at load, before source discovery or any warehouse or
provider access -- the same point a malformed `stel_project.yml` fails.

Parsing needs rdflib (`stel[rdf]`), imported only when a project declares an
ontology. rdflib does not follow `owl:imports`, and Python's XML parser does
not resolve external entities, so a parse reads exactly the one validated file.
Parser errors are reduced to a line number: their text can quote the file.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from pydantic import ValidationError

from .config.class_hierarchy import ClassDef, class_names
from .config.identifiers import validate_node_name
from .config.loader import (
    ConfigError,
    _reject_symlink_directory_components,
    _validate_config_file,
)
from .config.ontology import OntologyImport, OntologyInclude
from .config.vocabulary import RelationTypeDef, Vocabulary, VocabularyTerm
from .optional_dependencies import OptionalDependencyError, import_optional_dependency
from .paths import resolve_within_project

if TYPE_CHECKING:
    from rdflib import Graph
    from rdflib.term import Node, URIRef

    from .config.project import ProjectConfig

_TERM = "term"
_CLASS = "class"
_RELATION = "relation"
_KIND_LABELS = {
    _TERM: "a skos:Concept or owl:NamedIndividual",
    _CLASS: "an owl:Class",
    _RELATION: "an owl:ObjectProperty",
}
_LOCAL_NAME = re.compile(r"[^#/:]*$")
# What a relative IRI in an ontology file resolves against, followed by the
# file's project-relative path. Documented, so an allow-list can name one.
RELATIVE_IRI_BASE = "https://stel.invalid/"


def resolve_ontologies(project: ProjectConfig, project_dir: Path) -> None:
    """Merge every `ontologies:` import into the project, then validate it whole."""
    if not project.ontologies:
        return
    imported_vocabularies: dict[str, Vocabulary] = {}
    imported_classes: list[ClassDef] = []
    imported_relations: list[RelationTypeDef] = []
    class_origin: dict[str, str] = {name: "`classes:`" for name in class_names(project.classes)}
    for ontology in project.ontologies:
        surface = f"Ontology '{ontology.path}'"
        graph = _parse(ontology, project_dir, surface)
        imported = _Importer(graph, ontology, surface)
        for class_def in imported.classes:
            if class_def.name in class_origin:
                raise ConfigError(
                    f"{surface}: class '{class_def.name}' ({class_def.iri}) is also "
                    f"declared by {class_origin[class_def.name]}; give the "
                    "`include:` entry a different `name:`"
                )
            class_origin[class_def.name] = surface
        imported_classes.extend(imported.classes)
        imported_relations.extend(imported.relations)
        if ontology.vocabulary is None:
            continue
        name = ontology.vocabulary
        if name in project.vocabularies or name in imported_vocabularies:
            raise ConfigError(
                f"{surface}: vocabulary '{name}' is declared more than once; an "
                "ontology import owns its whole vocabulary"
            )
        imported_vocabularies[name] = imported.vocabulary()
    project.vocabularies.update(imported_vocabularies)
    project.classes = (*project.classes, *imported_classes)
    project.relations = (*project.relations, *imported_relations)
    try:
        project.validate_merged_declaration()
    except ValueError as exc:
        raise ConfigError(project.format_yaml_diagnostic(str(exc))) from None


def _parse(ontology: OntologyImport, project_dir: Path, surface: str) -> Graph:
    path = resolve_within_project(ontology.path, project_dir, surface=surface)
    configured = project_dir / ontology.path
    _reject_symlink_directory_components(configured, project_dir, description=surface)
    _validate_config_file(configured, allowed_root=project_dir, description=surface)
    try:
        rdflib = import_optional_dependency(
            "rdflib", extra="rdf", feature="Ontology import (`ontologies:`)"
        )
    except OptionalDependencyError as exc:
        # A declared feature whose extra is absent is a setup error, so it
        # crosses the same boundary as every other load failure: every CLI
        # command reports the install command and exits 2.
        raise ConfigError(str(exc)) from exc
    graph = rdflib.Graph()
    data = path.read_bytes()
    # A fixed base, not the file's location: rdflib would otherwise resolve a
    # relative IRI against this machine's absolute path, putting it into
    # identities and artifacts. Hierarchical, because rdflib leaves a relative
    # reference unresolved against a `urn:` base, and under the reserved
    # `.invalid` domain (RFC 2606), so it can never name a real resource. The
    # path is the validated project-relative one, so an absolute `path:` inside
    # the project yields the same base as its relative spelling.
    relative = path.relative_to(project_dir.resolve())
    base = f"{RELATIVE_IRI_BASE}{PurePosixPath(relative).as_posix()}"
    try:
        graph.parse(data=data, format=ontology.parser_format(), publicID=base)
    except Exception as exc:  # rdflib raises a different type per parser
        line = _error_line(exc)
        where = f" near line {line}" if line is not None else ""
        raise ConfigError(
            f"{surface}: cannot parse the file as {ontology.parser_format()}{where}"
        ) from None
    return graph


def _error_line(exc: Exception) -> int | None:
    """The line a parser error points at, without the error's own text."""
    get_line: Callable[[], object] | None = getattr(exc, "getLineNumber", None)
    line = get_line() if callable(get_line) else getattr(exc, "lines", None)
    return line if isinstance(line, int) and not isinstance(line, bool) else None


class _Importer:
    """One ontology file's allow-list, mapped onto stel's declaration."""

    def __init__(self, graph: Graph, ontology: OntologyImport, surface: str) -> None:
        # Imported lazily with the parser; the namespaces are rdflib's own.
        from rdflib.namespace import OWL, RDF, RDFS, SKOS

        self._graph = graph
        self._ontology = ontology
        self._surface = surface
        self._rdf, self._rdfs, self._owl, self._skos = RDF, RDFS, OWL, SKOS
        self._kinds = {entry.iri: self._kind(entry) for entry in ontology.include}
        self._class_names = {
            entry.iri: self._class_name(entry)
            for entry in ontology.include
            if self._kinds[entry.iri] == _CLASS
        }
        term_entries = [e for e in ontology.include if self._kinds[e.iri] == _TERM]
        if term_entries and ontology.vocabulary is None:
            raise self._error(
                f"`include:` names {term_entries[0].iri}, {_KIND_LABELS[_TERM]}, but "
                "the entry declares no `vocabulary:` to put terms in"
            )
        if ontology.vocabulary is not None and not term_entries:
            raise self._error(
                f"declares `vocabulary: {ontology.vocabulary}`, but its `include:` "
                f"names no {_KIND_LABELS[_TERM]}"
            )
        self._labels = {entry.iri: self._term_label(entry) for entry in term_entries}
        self._reject_collisions(self._class_names, "class name")
        self._reject_collisions(self._labels, "term label")
        self.classes = [
            self._class(entry) for entry in ontology.include if entry.iri in self._class_names
        ]
        self.terms = [self._term(entry) for entry in term_entries]
        self.relations = [
            pairing
            for entry in ontology.include
            if self._kinds[entry.iri] == _RELATION
            for pairing in self._relation(entry)
        ]

    def vocabulary(self) -> Vocabulary:
        try:
            return Vocabulary(terms=self.terms)
        except ValidationError as exc:
            # Messages here are the declaration's own (a cyclic `broader` chain
            # names a label), never a dump of parser input.
            message = exc.errors()[0]["msg"].removeprefix("Value error, ")
            raise self._error(message) from None

    # -- classification ---------------------------------------------------

    def _kind(self, entry: OntologyInclude) -> str:
        iri = self._iri(entry.iri)
        types = set(self._graph.objects(iri, self._rdf.type))
        kinds = set()
        if types & {self._skos.Concept, self._owl.NamedIndividual}:
            kinds.add(_TERM)
        if types & {self._owl.Class, self._rdfs.Class}:
            kinds.add(_CLASS)
        if self._owl.ObjectProperty in types:
            kinds.add(_RELATION)
        if len(kinds) == 1:
            return kinds.pop()
        if kinds:
            described = " and ".join(_KIND_LABELS[kind] for kind in sorted(kinds))
            raise self._error(
                f"{entry.iri} is typed as {described}; stel imports each IRI as one kind"
            )
        if (iri, None, None) in self._graph:
            raise self._error(
                f"{entry.iri} is in the file but is not typed as "
                f"{_KIND_LABELS[_TERM]}, {_KIND_LABELS[_CLASS]} or "
                f"{_KIND_LABELS[_RELATION]}"
            )
        raise self._error(
            f"allow-listed IRI {entry.iri} is not declared in the file (prefixes are "
            "not expanded; check the full IRI)"
        )

    # -- classes ----------------------------------------------------------

    def _class_name(self, entry: OntologyInclude) -> str:
        name = entry.name if entry.name is not None else _local_name(entry.iri)
        try:
            return validate_node_name(name, kind="Class")
        except ValueError:
            raise self._error(
                f"class {entry.iri} would be named {name!r}, which is not a valid class "
                "name; give the `include:` entry a `name:`"
            ) from None

    def _class(self, entry: OntologyInclude) -> ClassDef:
        iri = self._iri(entry.iri)
        parents = self._allow_listed(
            self._graph.objects(iri, self._rdfs.subClassOf), self._class_names
        )
        if len(parents) > 1:
            raise self._error(
                f"class {entry.iri} is a subclass of {len(parents)} allow-listed classes "
                f"({', '.join(parents)}); a stel class has one parent, so leave all "
                "but one out of `include:`"
            )
        return ClassDef(
            name=self._class_names[entry.iri],
            subclass_of=self._class_names[parents[0]] if parents else None,
            iri=entry.iri,
        )

    # -- terms ------------------------------------------------------------

    def _term_label(self, entry: OntologyInclude) -> str:
        if entry.name is not None:
            return entry.name
        iri = self._iri(entry.iri)
        labels = self._literals(iri, self._skos.prefLabel) or self._literals(
            iri, self._rdfs.label
        )
        if not labels:
            raise self._error(
                f"term {entry.iri} has no skos:prefLabel or rdfs:label in "
                f"'{self._ontology.lang}' or without a language tag; give the "
                "`include:` entry a `name:`"
            )
        return self._single(labels, entry.iri, "label")

    def _term(self, entry: OntologyInclude) -> VocabularyTerm:
        iri = self._iri(entry.iri)
        label = self._labels[entry.iri]
        definitions = self._literals(iri, self._skos.definition)
        broader = self._allow_listed(
            [
                *self._graph.objects(iri, self._skos.broader),
                *self._graph.subjects(self._skos.narrower, iri),
            ],
            self._labels,
        )
        if len(broader) > 1:
            raise self._error(
                f"term {entry.iri} has {len(broader)} allow-listed broader terms "
                f"({', '.join(broader)}); a stel term has one, so leave all but one "
                "out of `include:`"
            )
        classes = self._allow_listed(self._graph.objects(iri, self._rdf.type), self._class_names)
        if len(classes) > 1:
            raise self._error(
                f"term {entry.iri} is an instance of {len(classes)} allow-listed classes "
                f"({', '.join(classes)}); a stel term has one `class:`"
            )
        aliases = [alias for alias in self._literals(iri, self._skos.altLabel) if alias != label]
        return VocabularyTerm.model_validate(
            {
                "label": label,
                "description": self._single(definitions, entry.iri, "definition")
                if definitions
                else None,
                "aliases": aliases,
                "broader": self._labels[broader[0]] if broader else None,
                "class": self._class_names[classes[0]] if classes else None,
                "iri": entry.iri,
            }
        )

    # -- relations --------------------------------------------------------

    def _relation(self, entry: OntologyInclude) -> list[RelationTypeDef]:
        iri = self._iri(entry.iri)
        name = entry.name if entry.name is not None else _local_name(entry.iri)
        if not name:
            raise self._error(
                f"relation {entry.iri} has no local name; give the `include:` entry a `name:`"
            )
        ends: list[list[str]] = []
        for predicate, role in ((self._rdfs.domain, "domain"), (self._rdfs.range, "range")):
            members = self._class_members(self._graph.objects(iri, predicate))
            if not members:
                raise self._error(
                    f"relation {entry.iri} declares no named rdfs:{role}; stel needs a "
                    "domain and a range class to declare a relation pairing"
                )
            allowed = self._allow_listed(members, self._class_names)
            if not allowed:
                raise self._error(
                    f"relation {entry.iri} has no rdfs:{role} class in `include:`; add "
                    f"one of {', '.join(sorted(str(member) for member in members))}"
                )
            ends.append(allowed)
        # owl:FunctionalProperty is ADR-0056's `functional`: at most one object
        # per subject. Read from the file so a hand-written pairing of the same
        # name has something to agree with.
        functional = (iri, self._rdf.type, self._owl.FunctionalProperty) in self._graph
        return [
            RelationTypeDef(
                name=name,
                subject_class=self._class_names[subject],
                object_class=self._class_names[obj],
                functional=functional,
                iri=entry.iri,
            )
            for subject in ends[0]
            for obj in ends[1]
        ]

    def _class_members(self, nodes: Iterable[Node]) -> list[Node]:
        """Named classes, with an `owl:unionOf` blank node expanded to its members."""
        from rdflib.collection import Collection
        from rdflib.term import BNode, URIRef

        members: list[Node] = []
        for node in nodes:
            if isinstance(node, URIRef):
                members.append(node)
                continue
            union = self._graph.value(node, self._owl.unionOf) if isinstance(node, BNode) else None
            if union is not None:
                members.extend(m for m in Collection(self._graph, union) if isinstance(m, URIRef))
        return members

    # -- helpers ----------------------------------------------------------

    def _iri(self, value: str) -> URIRef:
        from rdflib.term import URIRef

        return URIRef(value)

    def _allow_listed(self, nodes: Iterable[Node], allowed: dict[str, str]) -> list[str]:
        """The allow-listed IRIs among `nodes`, in allow-list order, deduplicated."""
        present = {str(node) for node in nodes}
        return [iri for iri in allowed if iri in present]

    def _literals(self, iri: URIRef, predicate: Node) -> list[str]:
        """Literal values in the configured language, else untagged; sorted."""
        from rdflib.term import Literal

        values = [o for o in self._graph.objects(iri, predicate) if isinstance(o, Literal)]
        lang = self._ontology.lang.lower()
        tagged = [v for v in values if v.language is not None and v.language.lower() == lang]
        chosen = tagged or [v for v in values if v.language is None]
        return sorted({str(value).strip() for value in chosen} - {""})

    def _single(self, values: Sequence[str], iri: str, what: str) -> str:
        if len(values) > 1:
            raise self._error(
                f"term {iri} has {len(values)} candidate {what}s in "
                f"'{self._ontology.lang}'; stel needs exactly one"
            )
        return values[0]

    def _reject_collisions(self, names: dict[str, str], what: str) -> None:
        by_name: dict[str, str] = {}
        for iri, name in names.items():
            if name in by_name:
                raise self._error(
                    f"{by_name[name]} and {iri} both import as {what} '{name}'; give one "
                    "of the `include:` entries a `name:`"
                )
            by_name[name] = iri

    def _error(self, detail: str) -> ConfigError:
        return ConfigError(f"{self._surface}: {detail}")


def _local_name(iri: str) -> str:
    match = _LOCAL_NAME.search(iri)
    return match.group(0) if match else ""

