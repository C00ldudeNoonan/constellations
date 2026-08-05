"""A rich, realistic demo bundle for the concept cloud (#255).

Unlike the tiny `placeholder_export()` (a minimal fixture for tests), this is a
sizable economic-news dataset — ~45 canonical entities across eight spaCy-style
types, co-occurrence and directed relation edges, and a layered dbt DAG — so the
visualization's value reads clearly: dense frequency-sized clusters, colored by
entity type, tied down to the models that produced them. It is entirely
synthetic/curated; no real project data.
"""
from __future__ import annotations

import re
from typing import cast

from .schema import (
    Concept,
    ConceptCloudExport,
    ConceptEdge,
    CrossLayerEdge,
    DagEdge,
    DagNode,
    DagPlane,
    LinkStatus,
    Provenance,
)

_LINKING_NODE = "model.economic_data.link_entities"

# (display, spaCy label, mention frequency, link_status)
_CONCEPTS: tuple[tuple[str, str, int, str], ...] = (
    ("Federal Reserve", "ORG", 142, "matched"),
    ("U.S. Treasury", "ORG", 74, "matched"),
    ("European Central Bank", "ORG", 61, "matched"),
    ("Bank of Japan", "ORG", 33, "matched"),
    ("Bank of England", "ORG", 29, "matched"),
    ("International Monetary Fund", "ORG", 41, "matched"),
    ("World Bank", "ORG", 24, "matched"),
    ("Securities and Exchange Commission", "ORG", 22, "ambiguous"),
    ("JPMorgan Chase", "ORG", 38, "matched"),
    ("Goldman Sachs", "ORG", 31, "matched"),
    ("Bank of America", "ORG", 19, "matched"),
    ("BlackRock", "ORG", 17, "matched"),
    ("Berkshire Hathaway", "ORG", 21, "matched"),
    ("OPEC", "ORG", 26, "matched"),
    ("Apple", "ORG", 34, "ambiguous"),
    ("Microsoft", "ORG", 28, "matched"),
    ("Nvidia", "ORG", 30, "matched"),
    ("Tesla", "ORG", 23, "matched"),
    ("Congress", "ORG", 57, "matched"),
    ("Senate", "ORG", 25, "matched"),
    ("Jerome Powell", "PERSON", 98, "matched"),
    ("Janet Yellen", "PERSON", 63, "matched"),
    ("Christine Lagarde", "PERSON", 44, "matched"),
    ("Kazuo Ueda", "PERSON", 18, "matched"),
    ("Andrew Bailey", "PERSON", 15, "matched"),
    ("Warren Buffett", "PERSON", 27, "matched"),
    ("Jamie Dimon", "PERSON", 22, "matched"),
    ("Joe Biden", "PERSON", 71, "matched"),
    ("United States", "GPE", 131, "matched"),
    ("China", "GPE", 92, "matched"),
    ("Eurozone", "GPE", 48, "matched"),
    ("Japan", "GPE", 39, "matched"),
    ("United Kingdom", "GPE", 34, "matched"),
    ("Germany", "GPE", 30, "matched"),
    ("India", "GPE", 27, "matched"),
    ("$1.9 trillion", "MONEY", 20, "matched"),
    ("$500 billion", "MONEY", 16, "matched"),
    ("$25 billion", "MONEY", 12, "matched"),
    ("5.25%", "PERCENT", 45, "matched"),
    ("2%", "PERCENT", 52, "matched"),
    ("3.4%", "PERCENT", 24, "matched"),
    ("2008 financial crisis", "EVENT", 36, "matched"),
    ("COVID-19 pandemic", "EVENT", 43, "matched"),
    ("Dodd-Frank Act", "LAW", 19, "matched"),
    ("Basel III", "LAW", 14, "matched"),
    ("Republicans", "NORP", 40, "matched"),
    ("Democrats", "NORP", 42, "matched"),
)

# Undirected co-occurrence edges: (a, b, weight).
_COOCCURRENCE: tuple[tuple[str, str, int], ...] = (
    ("Federal Reserve", "Jerome Powell", 71),
    ("Federal Reserve", "5.25%", 44),
    ("Federal Reserve", "2%", 39),
    ("Federal Reserve", "United States", 58),
    ("Federal Reserve", "Congress", 22),
    ("Jerome Powell", "Congress", 18),
    ("Jerome Powell", "United States", 33),
    ("European Central Bank", "Christine Lagarde", 33),
    ("European Central Bank", "Eurozone", 40),
    ("European Central Bank", "Germany", 17),
    ("Bank of Japan", "Kazuo Ueda", 14),
    ("Bank of Japan", "Japan", 26),
    ("Bank of England", "Andrew Bailey", 12),
    ("Bank of England", "United Kingdom", 21),
    ("U.S. Treasury", "Janet Yellen", 41),
    ("U.S. Treasury", "United States", 44),
    ("U.S. Treasury", "$1.9 trillion", 13),
    ("International Monetary Fund", "World Bank", 19),
    ("International Monetary Fund", "United States", 14),
    ("JPMorgan Chase", "Jamie Dimon", 20),
    ("JPMorgan Chase", "Goldman Sachs", 12),
    ("Goldman Sachs", "BlackRock", 9),
    ("Berkshire Hathaway", "Warren Buffett", 21),
    ("United States", "China", 66),
    ("China", "India", 18),
    ("Securities and Exchange Commission", "Dodd-Frank Act", 13),
    ("Securities and Exchange Commission", "Congress", 11),
    ("2008 financial crisis", "Basel III", 12),
    ("2008 financial crisis", "Dodd-Frank Act", 15),
    ("COVID-19 pandemic", "$1.9 trillion", 17),
    ("Apple", "Microsoft", 16),
    ("Microsoft", "Nvidia", 14),
    ("Nvidia", "Tesla", 12),
    ("Apple", "Nvidia", 11),
    ("OPEC", "$500 billion", 8),
    ("Republicans", "Congress", 28),
    ("Democrats", "Congress", 30),
    ("Republicans", "Democrats", 34),
    ("Joe Biden", "United States", 52),
    ("Joe Biden", "Democrats", 24),
    ("Congress", "Senate", 19),
)

# Directed, higher-confidence assertions: (subject, object, relation, confidence).
_ASSERTIONS: tuple[tuple[str, str, str, float], ...] = (
    ("Jerome Powell", "Federal Reserve", "chairs", 0.97),
    ("Christine Lagarde", "European Central Bank", "leads", 0.95),
    ("Warren Buffett", "Berkshire Hathaway", "leads", 0.96),
    ("Jamie Dimon", "JPMorgan Chase", "leads", 0.94),
    ("Kazuo Ueda", "Bank of Japan", "leads", 0.93),
    ("Andrew Bailey", "Bank of England", "leads", 0.92),
    ("Janet Yellen", "U.S. Treasury", "leads", 0.90),
    ("Joe Biden", "United States", "president_of", 0.91),
)

# A layered dbt DAG the concepts sit above: (unique_id, label, resource_type).
_DAG_NODES: tuple[tuple[str, str, str], ...] = (
    ("source.economic_data.fred_series", "fred_series", "source"),
    ("source.economic_data.sec_filings", "sec_filings", "source"),
    ("source.economic_data.news_articles", "news_articles", "source"),
    ("source.economic_data.bls_releases", "bls_releases", "source"),
    ("model.economic_data.stg_news_articles", "stg_news_articles", "model"),
    ("model.economic_data.stg_sec_filings", "stg_sec_filings", "model"),
    ("model.economic_data.stg_fred_series", "stg_fred_series", "model"),
    ("model.economic_data.nlp_entities", "nlp_entities", "model"),
    (_LINKING_NODE, "link_entities", "model"),
    ("model.economic_data.extract_relations", "extract_relations", "model"),
    ("model.economic_data.mart_entity_network", "mart_entity_network", "model"),
    ("model.economic_data.mart_indicator_trends", "mart_indicator_trends", "model"),
    ("exposure.economic_data.macro_dashboard", "macro_dashboard", "exposure"),
)

_DAG_EDGES: tuple[tuple[str, str], ...] = (
    ("source.economic_data.news_articles", "model.economic_data.stg_news_articles"),
    ("source.economic_data.sec_filings", "model.economic_data.stg_sec_filings"),
    ("source.economic_data.fred_series", "model.economic_data.stg_fred_series"),
    ("model.economic_data.stg_news_articles", "model.economic_data.nlp_entities"),
    ("model.economic_data.stg_sec_filings", "model.economic_data.nlp_entities"),
    ("model.economic_data.nlp_entities", _LINKING_NODE),
    (_LINKING_NODE, "model.economic_data.extract_relations"),
    (_LINKING_NODE, "model.economic_data.mart_entity_network"),
    ("model.economic_data.extract_relations", "model.economic_data.mart_entity_network"),
    ("model.economic_data.stg_fred_series", "model.economic_data.mart_indicator_trends"),
    ("model.economic_data.mart_entity_network", "exposure.economic_data.macro_dashboard"),
    ("model.economic_data.mart_indicator_trends", "exposure.economic_data.macro_dashboard"),
)


def _slug(display: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", display.lower()).strip("_")


def _canonical_id(display: str, label: str) -> str:
    return f"{label.lower()}:{_slug(display)}"


def demo_export() -> ConceptCloudExport:
    """A sizable, realistic economic-news concept cloud over a layered dbt DAG."""
    id_by_display = {d: _canonical_id(d, lbl) for d, lbl, _, _ in _CONCEPTS}
    documents_of = {d: max(1, freq // 3) for d, _, freq, _ in _CONCEPTS}

    concepts = tuple(
        Concept(
            canonical_id=id_by_display[display],
            display=display,
            label=label,
            frequency=freq,
            link_status=cast(LinkStatus, status),
            match_score=0.7 if status == "ambiguous" else None,
            provenance=Provenance(
                model="link_entities",
                source_node="source.dbt_ml_economic_data.link_entities",
                documents=documents_of[display],
            ),
        )
        for display, label, freq, status in _CONCEPTS
    )

    edges = [
        ConceptEdge(
            source=id_by_display[a], target=id_by_display[b],
            relation_type="co_occurs_with", method="co_occurrence",
            directed=False, weight=w,
        )
        for a, b, w in _COOCCURRENCE
    ] + [
        ConceptEdge(
            source=id_by_display[s], target=id_by_display[o],
            relation_type=rel, method="model_assertion", directed=True,
            weight=max(3, int(conf * 10)), confidence=conf,
        )
        for s, o, rel, conf in _ASSERTIONS
    ]

    dag_plane = DagPlane(
        nodes=tuple(
            DagNode(id=uid, label=label, resource_type=rt)
            for uid, label, rt in _DAG_NODES
        ),
        edges=tuple(
            DagEdge.model_validate({"from": a, "to": b}) for a, b in _DAG_EDGES
        ),
    )
    cross_layer_edges = tuple(
        CrossLayerEdge(concept=c.canonical_id, dag_node=_LINKING_NODE)
        for c in concepts
    )

    return ConceptCloudExport(
        generated_at="2026-08-05T00:00:00Z",
        project="economic_data",
        dag_plane=dag_plane,
        concepts=concepts,
        concept_edges=tuple(edges),
        cross_layer_edges=cross_layer_edges,
    )
