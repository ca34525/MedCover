"""Package-bundled DuckDB SQL definitions loaded via importlib.resources."""

from importlib.resources import files


def load_sql(filename: str) -> str:
    """Read a packaged SQL file from ``medcover.sql``.

    Inputs: SQL filename relative to ``src/medcover/sql/`` (for example ``silver_policies.sql``).
    Output grain: single UTF-8 SQL query string.
    Invariants: works identically for editable installs, built wheels, and Docker images.
    """
    resource = files("medcover.sql").joinpath(filename)
    return resource.read_text(encoding="utf-8")
