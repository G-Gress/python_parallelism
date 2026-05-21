"""
python_parallelism package.
Contains modules for experimenting with different parallelization strategies.
"""

from __future__ import annotations

try:
	from importlib.metadata import version as _version
except Exception:  # pragma: no cover
	_version = None


def __getattr__(name: str):
	if name == "__version__":
		if _version is None:
			return "0.0.0"
		return _version("python_parallelism")
	raise AttributeError(name)
