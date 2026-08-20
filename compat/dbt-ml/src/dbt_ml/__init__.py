"""`dbt-ml` was renamed to Constellations, installed and invoked as `stel`.

This package exists only so that someone who installs `dbt-ml` — from an old
pin, a stale bookmark, or the PyPI listing — finds out where the project went
instead of getting a version frozen at 0.8.0 with no explanation.

It is not a compatibility layer. It deliberately does not re-export stel's API
or alias its submodules: a shim that made `import dbt_ml.adapters` keep working
would hide the rename behind a facade that then has to be maintained and
eventually removed. `stel` is a hard dependency, so upgrading gets you the real
package and the fix is a one-word change to the import.
"""

from __future__ import annotations

import sys
import warnings

__all__ = ["MESSAGE", "__version__"]

__version__ = "0.8.1"

MESSAGE = (
    "dbt-ml has been renamed to Constellations and is published as `stel`. "
    "This package is a redirect and carries no functionality. Install `stel` "
    "(already pulled in as a dependency) and change `import dbt_ml` to "
    "`import stel`. Project files are now `stel_project.yml`, profiles live in "
    "`~/.stel/`, environment variables use the `STEL_` prefix, and the CLI is "
    "`stel`. See https://github.com/C00ldudeNoonan/constellations."
)

warnings.warn(MESSAGE, DeprecationWarning, stacklevel=2)


def _redirect_cli() -> int:
    """Stand in for the old `dbt-ml` console script.

    Exits non-zero so a script or CI job that still shells out to `dbt-ml`
    fails visibly rather than appearing to succeed at doing nothing.
    """
    print(MESSAGE, file=sys.stderr)
    print("\nRun `stel` instead. `stel --help` lists the commands.", file=sys.stderr)
    return 1
