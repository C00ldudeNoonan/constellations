from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dbt-ml")
except PackageNotFoundError:
    __version__ = "unknown"
