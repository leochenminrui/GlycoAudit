"""Project path configuration for GlycoAudit.

All paths are resolved relative to the installed source tree by default.  A
different checkout or data root can be selected with ``GLYCOAUDIT_ROOT``.
Keeping path resolution in one module prevents analysis scripts from depending
on an author's workstation layout.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the GlycoAudit repository root.

    ``GLYCOAUDIT_ROOT`` may point to an alternative checkout or a mounted data
    release.  Relative overrides are resolved against the current directory.
    """

    override = os.environ.get("GLYCOAUDIT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    """Return the repository data directory."""

    return get_project_root() / "data"


def get_outputs_dir() -> Path:
    """Return the directory for newly generated, non-release outputs."""

    return get_project_root() / "outputs"
