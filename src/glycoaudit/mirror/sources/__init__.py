"""Source connectors for the glyco mirror."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..config import MirrorConfig, SourceConfig
from ..io import DownloadRecord, HTTPClient


@dataclass
class SourceItem:
    """Represents an item to be downloaded from a source."""

    source: str
    item_id: str
    url: str
    item_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSource(ABC):
    """Base class for source connectors."""

    SOURCE_NAME: str = ""
    CATEGORY: str = ""

    def __init__(self, config: MirrorConfig, http_client: HTTPClient):
        """Initialize the source connector.

        Args:
            config: Mirror configuration.
            http_client: HTTP client for making requests.
        """
        self.config = config
        self.http = http_client
        self.source_config = config.sources.get(self.SOURCE_NAME, SourceConfig())

    @property
    def data_dir(self) -> Path:
        """Get the data directory for this source."""
        return self.config.get_source_dir(self.SOURCE_NAME)

    @property
    def is_enabled(self) -> bool:
        """Check if this source is enabled."""
        return self.config.is_source_enabled(self.SOURCE_NAME)

    @abstractmethod
    def list_items(self) -> Iterator[SourceItem]:
        """List all items available for download.

        Yields:
            SourceItem instances.
        """
        pass

    @abstractmethod
    def download_item(self, item: SourceItem) -> list[DownloadRecord]:
        """Download an item.

        Args:
            item: Item to download.

        Returns:
            List of download records (may be multiple files per item).
        """
        pass

    def parse_metadata(
        self, item: SourceItem, local_paths: list[Path]
    ) -> dict[str, Any]:
        """Parse metadata from downloaded item.

        Args:
            item: Source item.
            local_paths: Paths to downloaded files.

        Returns:
            Metadata dictionary.
        """
        return {
            "source": self.SOURCE_NAME,
            "item_id": item.item_id,
            "item_type": item.item_type,
            "url": item.url,
            "local_paths": [str(p) for p in local_paths],
            **item.metadata,
        }

    def requires_auth(self) -> bool:
        """Check if this source requires authentication."""
        return False

    def auth_instructions(self) -> str:
        """Get authentication instructions."""
        return ""


# Import source implementations
from .cfg_array import CFGArraySource
from .glycam_web import GlycamWebSource
from .glycopost import GlycoPostSource
from .glycoshape import GlycoShapeSource
from .glygen import GlyGenSource
from .glytoucan import GlyTouCanSource
from .pride import PrideSource
from .sugarbind_rdf import SugarBindRDFSource

# Registry of all sources
SOURCES: dict[str, type[BaseSource]] = {
    "glytoucan": GlyTouCanSource,
    "glycoshape": GlycoShapeSource,
    "glycam_web": GlycamWebSource,
    "cfg": CFGArraySource,
    "glygen": GlyGenSource,
    "glycopost": GlycoPostSource,
    "pride": PrideSource,
    "sugarbind": SugarBindRDFSource,
}


def get_source(
    source_name: str, config: MirrorConfig, http_client: HTTPClient
) -> BaseSource:
    """Get a source connector instance.

    Args:
        source_name: Name of the source.
        config: Mirror configuration.
        http_client: HTTP client.

    Returns:
        Source connector instance.

    Raises:
        ValueError: If source is not found.
    """
    if source_name not in SOURCES:
        raise ValueError(f"Unknown source: {source_name}")

    return SOURCES[source_name](config, http_client)


def get_all_sources(
    config: MirrorConfig, http_client: HTTPClient
) -> list[BaseSource]:
    """Get all source connector instances.

    Args:
        config: Mirror configuration.
        http_client: HTTP client.

    Returns:
        List of source connector instances.
    """
    return [
        get_source(name, config, http_client)
        for name in SOURCES
        if config.is_source_enabled(name)
    ]
