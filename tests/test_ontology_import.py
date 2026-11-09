"""Importing a SKOS/OWL ontology into the domain declaration (issue #767 / ALE-419)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import rdflib
import yaml
from click.testing import CliRunner

from stel.cli import cli
from stel.config.class_hierarchy import ClassDef, class_names
from stel.config.loader import ConfigError, load_project
from stel.config.ontology import OntologyImport
from stel.config.project import ProjectConfig
from stel.mcp_server.catalog import ArtifactCatalog

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ontology_import"
LP = "https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/LegalPersons/"
REL = "https://spec.edmcouncil.org/fibo/ontology/FND/Relations/Relations/"
CR = "https://www.omg.org/spec/LCC/Countries/CountryRepresentation/"
ISO = "https://www.omg.org/spec/LCC/Countries/ISO3166-1-CountryCodes/"
SECTOR = "https://example.com/sectors#"
PREFIXES = (
    "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
    "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .\n"
    "@prefix ex: <https://example.com/o#> .\n"
)
EX = "https://example.com/o#"


def _example(tmp_path: Path) -> Path:
    project = tmp_path / "ontology_import"
    shutil.copytree(EXAMPLE, project, ignore=shutil.ignore_patterns("target", "__pycache__"))
    return project


def _project(
    tmp_path: Path,
    turtle: str,
    ontology: dict[str, Any],
    **project: Any,
) -> Path:
    """A one-ontology project with no models: load is the whole subject."""
    (tmp_path / "onto.ttl").write_text(PREFIXES + turtle, encoding="utf-8")
    (tmp_path / "stel_project.yml").write_text(
        yaml.safe_dump(
            {
                "name": "onto",
                "duckdb": {"path": "warehouse.duckdb"},
                "ontologies": [{"path": "onto.ttl", **ontology}],
                **project,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_fibo_subset_imports_with_iris_hierarchy_and_allow_list_order(tmp_path: Path) -> None:
    project, _, models = load_project(_example(tmp_path))

    assert project.classes == (
        ClassDef(name="LegalPerson", iri=f"{LP}LegalPerson"),
        ClassDef(name="LegalEntity", subclass_of="LegalPerson", iri=f"{LP}LegalEntity"),
        ClassDef(name="Country", iri=f"{CR}Country"),
    )
    [relation] = project.relations
    assert (relation.name, relation.subject_class, relation.object_class, relation.iri) == (
        "isDomiciledIn",
        "LegalEntity",
        "Country",
        f"{REL}isDomiciledIn",
    )
    country = project.vocabularies["country"]
    # Allow-list order, not file order; Japan is in the file but not allowed;
    # the German-tagged label is not read under `lang: en`.
    assert country.labels() == ["United States of America", "Germany"]
    usa = country.terms[0]
    assert (usa.entity_class, usa.iri, usa.aliases) == (
        "Country",
        f"{ISO}UnitedStatesOfAmerica",
        ["USA", "United States"],
    )
    sector = project.vocabularies["sector"]
    assert sector.labels() == ["financials", "banks"]
    assert sector.terms[0].description == "Banks, insurers and other financial firms"
    assert sector.terms[1].broader == "financials"
    # Imported vocabularies drive enum fields like any other declaration.
    assert [field.values for field in models[0].fields] == [
        ["United States of America", "Germany"],
        ["financials", "banks"],
    ]


def test_rdf_xml_file_imports_the_same_declaration_as_turtle(tmp_path: Path) -> None:
    turtle_project = _example(tmp_path / "ttl")
    xml_project = _example(tmp_path / "xml")
    graph = rdflib.Graph().parse(turtle_project / "ontologies" / "fibo_subset.ttl")
    graph.serialize(xml_project / "ontologies" / "fibo_subset.owl", format="xml")
    config = xml_project / "stel_project.yml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("fibo_subset.ttl", "fibo_subset.owl"),
        encoding="utf-8",
    )

    from_turtle, _, _ = load_project(turtle_project)
    from_xml, _, _ = load_project(xml_project)

    assert from_xml.classes == from_turtle.classes
    assert from_xml.relations == from_turtle.relations
    assert from_xml.vocabularies == from_turtle.vocabularies


def test_allow_listed_iri_missing_from_the_file_fails_compile(tmp_path: Path) -> None:
    project = _example(tmp_path)
    config = project / "stel_project.yml"
    text = config.read_text(encoding="utf-8")
    config.write_text(text.replace(f"{ISO}Germany", f"{ISO}France"), encoding="utf-8")

    result = CliRunner().invoke(cli, ["compile", "--project-dir", str(project)])

    assert result.exit_code == 2
    assert f"allow-listed IRI {ISO}France is not declared in the file" in result.output
    assert not (project / "target").exists()


def test_hand_written_declaration_can_refer_to_imported_classes(tmp_path: Path) -> None:
    """Cross-reference checks wait for the import, then still run on the whole."""
    turtle = "ex:Company a owl:Class .\n"
    declared = {
        "classes": ["country", {"name": "bank", "subclass_of": "Company"}],
        "relations": [
            {"name": "located_in", "subject_class": "Company", "object_class": "country"}
        ],
        "vocabularies": {"banks": {"terms": [{"label": "Example Bank", "class": "Company"}]}},
    }
    project, _, _ = load_project(
        _project(tmp_path, turtle, {"include": [f"{EX}Company"]}, **declared)
    )
    assert [r.subject_class for r in project.relations] == ["Company"]

    declared["relations"][0]["object_class"] = "nation"
    _project(tmp_path, turtle, {"include": [f"{EX}Company"]}, **declared)
    with pytest.raises(ConfigError, match="`object_class: nation`, which is not declared"):
        load_project(tmp_path)


@pytest.mark.parametrize(
    ("turtle", "ontology", "project", "message"),
    [
        pytest.param(
            "ex:Thing rdfs:label 'thing' .\n",
            {"include": [f"{EX}Thing"]},
            {},
            "is in the file but is not typed",
            id="untyped",
        ),
        pytest.param(
            "ex:Both a owl:Class, skos:Concept ; skos:prefLabel 'both' .\n",
            {"include": [f"{EX}Both"], "vocabulary": "v"},
            {},
            "stel imports each IRI as one kind",
            id="punned",
        ),
        pytest.param(
            "ex:a a skos:Concept ; skos:prefLabel 'a' .\n",
            {"include": [f"{EX}a"]},
            {},
            "declares no `vocabulary:`",
            id="term-without-vocabulary",
        ),
        pytest.param(
            "ex:C a owl:Class .\n",
            {"include": [f"{EX}C"], "vocabulary": "v"},
            {},
            "names no a skos:Concept",
            id="vocabulary-without-terms",
        ),
        pytest.param(
            "ex:a a skos:Concept .\n",
            {"include": [f"{EX}a"], "vocabulary": "v"},
            {},
            "has no skos:prefLabel or rdfs:label",
            id="term-without-label",
        ),
        pytest.param(
            "ex:a a skos:Concept ; skos:prefLabel 'a', 'b' .\n",
            {"include": [f"{EX}a"], "vocabulary": "v"},
            {},
            "2 candidate labels",
            id="two-labels",
        ),
        pytest.param(
            "ex:a a skos:Concept ; skos:prefLabel 'x' .\n"
            "ex:b a skos:Concept ; skos:prefLabel 'x' .\n",
            {"include": [f"{EX}a", f"{EX}b"], "vocabulary": "v"},
            {},
            "both import as term label 'x'",
            id="label-collision",
        ),
        pytest.param(
            "ex:a a skos:Concept ; skos:prefLabel 'a' ; skos:broader ex:b .\n"
            "ex:b a skos:Concept ; skos:prefLabel 'b' ; skos:broader ex:a .\n",
            {"include": [f"{EX}a", f"{EX}b"], "vocabulary": "v"},
            {},
            "cyclic `broader` chain",
            id="broader-cycle",
        ),
        pytest.param(
            "ex:Legal-Person a owl:Class .\n",
            {"include": [f"{EX}Legal-Person"]},
            {},
            "'Legal-Person', which is not a valid class name",
            id="invalid-class-name",
        ),
        pytest.param(
            "ex:A a owl:Class .\nex:B a owl:Class .\nex:C a owl:Class ;"
            " rdfs:subClassOf ex:A, ex:B .\n",
            {"include": [f"{EX}A", f"{EX}B", f"{EX}C"]},
            {},
            "a stel class has one parent",
            id="two-parents",
        ),
        pytest.param(
            "ex:A a owl:Class .\nex:B a owl:Class .\n"
            "ex:p a owl:ObjectProperty ; rdfs:domain ex:A ; rdfs:range ex:B .\n",
            {"include": [f"{EX}A", f"{EX}p"]},
            {},
            f"has no rdfs:range class in `include:`; add one of {EX}B",
            id="range-not-allow-listed",
        ),
        pytest.param(
            "ex:A a owl:Class .\nex:p a owl:ObjectProperty ; rdfs:range ex:A .\n",
            {"include": [f"{EX}A", f"{EX}p"]},
            {},
            "declares no named rdfs:domain",
            id="no-domain",
        ),
        pytest.param(
            "ex:C a owl:Class .\n",
            {"include": [f"{EX}C"]},
            {"classes": ["C"]},
            "is also declared by `classes:`",
            id="class-collision",
        ),
        pytest.param(
            "ex:a a skos:Concept ; skos:prefLabel 'a' .\n",
            {"include": [f"{EX}a"], "vocabulary": "v"},
            {"vocabularies": {"v": {"terms": [{"label": "b"}]}}},
            "vocabulary 'v' is declared more than once",
            id="vocabulary-collision",
        ),
    ],
)
def test_ontology_that_cannot_map_onto_the_declaration_fails_load(
    tmp_path: Path,
    turtle: str,
    ontology: dict[str, Any],
    project: dict[str, Any],
    message: str,
) -> None:
    _project(tmp_path, turtle, ontology, **project)
    with pytest.raises(ConfigError) as raised:
        load_project(tmp_path)
    assert message in str(raised.value)


def test_name_override_resolves_a_collision_and_names_a_relation(tmp_path: Path) -> None:
    turtle = (
        "ex:Org a owl:Class .\n<https://example.com/other#Org> a owl:Class .\n"
        "ex:owns a owl:ObjectProperty ; rdfs:domain ex:Org ; rdfs:range"
        " [ owl:unionOf ( ex:Org <https://example.com/other#Org> ) ] .\n"
    )
    include = [
        f"{EX}Org",
        {"iri": "https://example.com/other#Org", "name": "OtherOrg"},
        {"iri": f"{EX}owns", "name": "owns_stake_in"},
    ]
    project, _, _ = load_project(_project(tmp_path, turtle, {"include": include}))
    assert class_names(project.classes) == ("Org", "OtherOrg")
    assert [(r.name, r.subject_class, r.object_class) for r in project.relations] == [
        ("owns_stake_in", "Org", "Org"),
        ("owns_stake_in", "Org", "OtherOrg"),
    ]


def test_functional_property_imports_as_a_functional_relation(tmp_path: Path) -> None:
    """A hand-written pairing of an imported relation must agree on `functional`
    (ADR-0056), so the import has to read it rather than default it to false."""
    turtle = (
        "ex:Company a owl:Class .\nex:Country a owl:Class .\nex:Person a owl:Class .\n"
        "ex:domiciledIn a owl:ObjectProperty, owl:FunctionalProperty ;"
        " rdfs:domain ex:Company ; rdfs:range ex:Country .\n"
    )
    include = [f"{EX}Company", f"{EX}Country", f"{EX}Person", f"{EX}domiciledIn"]
    extra = {"name": "domiciledIn", "subject_class": "Person", "object_class": "Country"}

    _project(tmp_path, turtle, {"include": include}, relations=[{**extra, "functional": True}])
    project, _, _ = load_project(tmp_path)
    assert [(r.subject_class, r.functional) for r in project.relations] == [
        ("Person", True),
        ("Company", True),
    ]

    _project(tmp_path, turtle, {"include": include}, relations=[extra])
    with pytest.raises(ConfigError, match="declared `functional` on one pairing and not"):
        load_project(tmp_path)


@pytest.mark.parametrize(
    ("ontology", "message"),
    [
        ({"path": "o.ttl", "include": []}, "at least 1 item"),
        ({"path": "o.ttl", "include": ["fibo:LegalEntity x"]}, "not an absolute IRI"),
        ({"path": "o.ttl", "include": ["LegalEntity"]}, "not an absolute IRI"),
        ({"path": "o.ttl", "include": [f"{EX}a", f"{EX}a"]}, "lists"),
        ({"path": "o.json", "include": [f"{EX}a"]}, "no recognized suffix"),
        ({"path": "o.ttl", "include": [f"{EX}a"], "vocabulary": "bad name"}, "invalid"),
        ({"path": "o.ttl", "include": [f"{EX}a"], "owl_imports": True}, "Extra inputs"),
    ],
)
def test_declaration_rejects_malformed_entries(ontology: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OntologyImport.model_validate(ontology)


def test_unparseable_file_reports_a_line_and_none_of_its_text(tmp_path: Path) -> None:
    _project(tmp_path, "ex:a a owl:Class .\nex:b SECRET-TEXT-IN-FILE\n", {"include": [f"{EX}a"]})
    with pytest.raises(ConfigError, match=r"cannot parse the file as turtle near line \d") as e:
        load_project(tmp_path)
    assert "SECRET-TEXT-IN-FILE" not in str(e.value)


def test_missing_rdf_extra_is_a_setup_error_naming_the_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`stel compile` used to exit 1 with no output: the import error escaped
    the configuration boundary (PR #773 review)."""
    _project(tmp_path, "ex:a a owl:Class .\n", {"include": [f"{EX}a"]})
    monkeypatch.setitem(sys.modules, "rdflib", None)

    result = CliRunner().invoke(cli, ["compile", "--project-dir", str(tmp_path)])

    assert result.exit_code == 2
    assert "pip install 'stel[rdf]'" in result.output


def test_project_without_ontologies_still_validates_at_construction() -> None:
    """Deferral is opt-in: an ordinary declaration fails as early as it always did."""
    with pytest.raises(ValueError, match="not declared under `classes:`"):
        ProjectConfig.model_validate(
            {
                "name": "p",
                "relations": [{"name": "r", "subject_class": "a", "object_class": "b"}],
            }
        )


@pytest.mark.parametrize("kind", ["escape", "symlink"])
def test_ontology_path_stays_inside_the_project_and_off_symlinks(
    tmp_path: Path, kind: str
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.ttl"
    outside.write_text(PREFIXES + "ex:a a owl:Class .\n", encoding="utf-8")
    path = "../outside.ttl"
    if kind == "symlink":
        (project / "linked.ttl").symlink_to(outside)
        path = "linked.ttl"
    (project / "stel_project.yml").write_text(
        yaml.safe_dump(
            {"name": "onto", "ontologies": [{"path": path, "include": [f"{EX}a"]}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"outside the project|symlink"):
        load_project(project)


@pytest.mark.parametrize("spelling", ["relative", "absolute"])
def test_rdf_xml_does_not_resolve_external_entities_or_local_paths(
    tmp_path: Path, spelling: str
) -> None:
    """A SYSTEM entity must not read a file, and a relative IRI must not take
    this machine's path as its base (it would reach identities and artifacts),
    even when `path:` is spelled absolute (PR #773 review)."""
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET-CONTENT", encoding="utf-8")
    (tmp_path / "onto.rdf").write_text(
        f"""<?xml version="1.0"?>
<!DOCTYPE rdf:RDF [ <!ENTITY xxe SYSTEM "file://{secret}"> ]>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:skos="http://www.w3.org/2004/02/skos/core#">
  <skos:Concept rdf:about="#a"><skos:prefLabel>term &xxe;</skos:prefLabel></skos:Concept>
</rdf:RDF>
""",
        encoding="utf-8",
    )
    (tmp_path / "stel_project.yml").write_text(
        yaml.safe_dump(
            {
                "name": "onto",
                "ontologies": [
                    {
                        "path": str(tmp_path / "onto.rdf")
                        if spelling == "absolute"
                        else "onto.rdf",
                        "vocabulary": "v",
                        "include": ["https://stel.invalid/onto.rdf#a"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    project, _, _ = load_project(tmp_path)
    [term] = project.vocabularies["v"].terms
    assert term.label == "term"
    assert str(tmp_path) not in json.dumps(term.model_dump())


def test_serving_manifest_carries_iris_and_the_catalog_reports_no_drift(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    project = _example(tmp_path)
    models = project / "models" / "classified.yml"
    model_file = yaml.safe_load(models.read_text(encoding="utf-8"))
    model_file["models"].append(
        {
            "name": "context",
            "depends_on": ["ref('classified_filings')"],
            "transform": {"type": "python", "module": "transforms.context"},
            "agent_context": {"contract": "agent_context/v1", "grain": "document_chunks"},
        }
    )
    models.write_text(yaml.safe_dump(model_file), encoding="utf-8")
    (project / "transforms").mkdir()
    (project / "transforms" / "context.py").write_text(
        "def run(inputs):\n    return inputs['classified_filings']\n", encoding="utf-8"
    )
    result = CliRunner().invoke(cli, ["compile", "--project-dir", str(project)])
    assert result.exit_code == 0, result.output

    catalog = ArtifactCatalog.load(project)

    assert catalog.vocabularies["country"].terms[1].iri == f"{ISO}Germany"
    assert catalog.vocabularies["sector"].terms[0].iri == f"{SECTOR}financials"
    assert catalog.class_hierarchy == {"LegalEntity": "LegalPerson"}
    manifest = json.loads((project / "target" / "manifest.json").read_text(encoding="utf-8"))
    assert {"name": "Country", "iri": f"{CR}Country"} in manifest[
        "declarations"
    ]["classes"]
    assert not [r for r in caplog.records if "differs" in r.getMessage()]
