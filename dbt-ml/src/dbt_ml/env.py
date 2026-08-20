"""Environment-variable names dbt-ml reads, and the single reader for them.

Every `DBT_ML_*` name the tool itself resolves is declared here so the set is
enumerable and a rename is one diff rather than a grep. Operator-chosen
credential variables are deliberately absent: dbt-ml never names those, it
resolves whatever `profiles.yml` references (see `credentials.py`), so they are
unaffected by anything in this module.

`read_env` takes several names on purpose. A name change that drops the old
variable fails silently — the value is simply absent and the caller falls back
to its default — so the reader supports reading a preferred name with older
names as fallbacks, making a future transition a one-argument change at each
call site instead of new branching logic.

Stdlib-only by design: config, logging, and the MCP authorization boundary all
import this, so it must stay a leaf.
"""

from __future__ import annotations

import os
from typing import overload

# Directory searched for profiles.yml before the project and home locations.
PROFILES_DIR_ENV = "DBT_ML_PROFILES_DIR"
# Truthy enables the stderr log handler and the live progress reporter.
VERBOSE_ENV = "DBT_ML_VERBOSE"
# Truthy attaches unsanitized provider error detail to local tracebacks.
PROVIDER_DEBUG_ENV = "DBT_ML_DEBUG_PROVIDER_ERRORS"

# MCP serving identity. These are security-relevant: when one stops resolving,
# a principal is still constructed — with no tenant and no access groups — so
# the failure surfaces as missing context rather than as an error.
MCP_PRINCIPAL_ID_ENV = "DBT_ML_MCP_PRINCIPAL_ID"
MCP_TENANT_ID_ENV = "DBT_ML_MCP_TENANT_ID"
MCP_ACCESS_GROUPS_ENV = "DBT_ML_MCP_ACCESS_GROUPS"
MCP_POLICY_CLAIMS_ENV = "DBT_ML_MCP_POLICY_CLAIMS"

# Every variable above, for enumeration by tests and diagnostics.
ENV_VARS: tuple[str, ...] = (
    PROFILES_DIR_ENV,
    VERBOSE_ENV,
    PROVIDER_DEBUG_ENV,
    MCP_PRINCIPAL_ID_ENV,
    MCP_TENANT_ID_ENV,
    MCP_ACCESS_GROUPS_ENV,
    MCP_POLICY_CLAIMS_ENV,
)


@overload
def read_env(*names: str) -> str | None: ...


@overload
def read_env(*names: str, default: str) -> str: ...


def read_env(*names: str, default: str | None = None) -> str | None:
    """Return the value of the first name present in the environment.

    Presence, not truthiness, decides: a variable explicitly set to the empty
    string shadows any later name, matching `os.environ.get` for a single name.
    """
    if not names:
        raise ValueError("read_env requires at least one variable name")
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default
