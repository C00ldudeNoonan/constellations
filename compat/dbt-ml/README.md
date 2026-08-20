# dbt-ml has been renamed

**This project is now [Constellations](https://github.com/C00ldudeNoonan/constellations), published on PyPI as [`stel`](https://pypi.org/project/stel/).**

```bash
pip install stel
```

`dbt-ml` borrowed another project's mark, read as an official dbt integration
when it is not one, and described the tool by what it sits next to rather than
what it does. The name changed at v0.9.0.

This `dbt-ml` release carries no functionality. It depends on `stel` and warns
on import, so an old pin resolves to something that tells you where the project
went. Version 0.8.0 remains on PyPI and still works — nothing was yanked, so
existing pins are unaffected.

## What changed

| Was | Now |
| --- | --- |
| `pip install dbt-ml` | `pip install stel` |
| `import dbt_ml` | `import stel` |
| `dbt-ml run` | `stel run` |
| `dbt_ml_project.yml` | `stel_project.yml` |
| `~/.dbt_ml/profiles.yml` | `~/.stel/profiles.yml` |
| `DBT_ML_*` environment variables | `STEL_*` |
| `dbt-ml[bigquery,...]` extras | `stel[bigquery,...]` |

Warehouse objects moved too — the state table, serving ledger and leases, and
the default schema. `stel migrate` renames them in place with rows preserved,
and stel refuses to run against an unmigrated warehouse rather than treating it
as a new project and reprocessing your corpus.

Values that were **not** changed, because they are written into places outside
the tool: the emitted dbt source name (`dbt_ml_<project>`), the `meta:`
namespace in generated `sources.yml`, fingerprint domains, and the classic-ML
artifact runtime key.

Full notes: [CHANGELOG](https://github.com/C00ldudeNoonan/constellations/blob/master/stel/CHANGELOG.md).
