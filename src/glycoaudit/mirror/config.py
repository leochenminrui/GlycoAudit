"""Configuration management for the glyco mirror."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class SourceConfig:
    """Configuration for a single data source."""

    enabled: bool = True
    mode: str = "api"
    note: str = ""
    # MS-specific options
    project_accessions: list[str] = field(default_factory=list)
    ms_download_policy: Literal["metadata_only", "small_files_only", "all"] = "metadata_only"
    ms_size_limit_mb: int = 100  # For small_files_only mode
    confirm_large_download: bool = False  # Must be True to download all MS files

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceConfig:
        return cls(
            enabled=data.get("enabled", True),
            mode=data.get("mode", "api"),
            note=data.get("note", ""),
            project_accessions=data.get("project_accessions", []),
            ms_download_policy=data.get("ms_download_policy", "metadata_only"),
            ms_size_limit_mb=data.get("ms_size_limit_mb", 100),
            confirm_large_download=data.get("confirm_large_download", False),
        )


@dataclass
class MirrorConfig:
    """Main configuration for the mirror."""

    root_dir: Path
    meta_dir: Path
    checksum_dir: Path
    log_dir: Path
    max_concurrent: int
    requests_per_second: float
    user_agent: str
    timeout_seconds: int
    max_retries: int
    sources: dict[str, SourceConfig]

    # Environment variable overrides
    glytoucan_api_key: str | None = None
    glycam_auth_token: str | None = None
    glycopost_api_key: str | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> MirrorConfig:
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Resolve paths relative to config file
        base_dir = config_path.parent.parent

        root_dir = Path(data.get("root_dir", "data/raw"))
        if not root_dir.is_absolute():
            root_dir = base_dir / root_dir

        meta_dir = Path(data.get("meta_dir", "data/meta"))
        if not meta_dir.is_absolute():
            meta_dir = base_dir / meta_dir

        checksum_dir = Path(data.get("checksum_dir", "data/checksums"))
        if not checksum_dir.is_absolute():
            checksum_dir = base_dir / checksum_dir

        log_dir = Path(data.get("log_dir", "logs"))
        if not log_dir.is_absolute():
            log_dir = base_dir / log_dir

        # Parse source configs
        sources = {}
        for name, source_data in data.get("sources", {}).items():
            if isinstance(source_data, dict):
                sources[name] = SourceConfig.from_dict(source_data)
            else:
                sources[name] = SourceConfig(enabled=bool(source_data))

        return cls(
            root_dir=root_dir,
            meta_dir=meta_dir,
            checksum_dir=checksum_dir,
            log_dir=log_dir,
            max_concurrent=data.get("max_concurrent", 4),
            requests_per_second=data.get("requests_per_second", 1.0),
            user_agent=data.get("user_agent", "public-glyco-mirror/1.0"),
            timeout_seconds=data.get("timeout_seconds", 60),
            max_retries=data.get("max_retries", 5),
            sources=sources,
            glytoucan_api_key=os.environ.get("GLYTOUCAN_API_KEY"),
            glycam_auth_token=os.environ.get("GLYCAM_AUTH_TOKEN"),
            glycopost_api_key=os.environ.get("GLYCOPOST_API_KEY"),
        )

    def ensure_directories(self) -> None:
        """Create all required directories."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.checksum_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.meta_dir / "source_manifests").mkdir(exist_ok=True)

        # Create source directories
        for source_name in self.sources:
            (self.root_dir / source_name).mkdir(exist_ok=True)

    def get_source_dir(self, source_name: str) -> Path:
        """Get the data directory for a specific source."""
        return self.root_dir / source_name

    def is_source_enabled(self, source_name: str) -> bool:
        """Check if a source is enabled."""
        if source_name not in self.sources:
            return False
        return self.sources[source_name].enabled
