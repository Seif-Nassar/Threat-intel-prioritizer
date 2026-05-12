from __future__ import annotations

import json
import logging
from pathlib import Path

from src.schema import AssetRecord

logger = logging.getLogger(__name__)

SAMPLE_ASSETS = [
    AssetRecord(asset_id="ASSET-001", name="Web Server (Prod)", asset_type="server",
                criticality=9.0, exposure=1.0, cpe="cpe:2.3:a:apache:http_server:2.4.51:*"),
    AssetRecord(asset_id="ASSET-002", name="Database Server", asset_type="server",
                criticality=10.0, exposure=0.2, cpe="cpe:2.3:a:mysql:mysql:8.0.30:*"),
    AssetRecord(asset_id="ASSET-003", name="Developer Workstation", asset_type="workstation",
                criticality=6.0, exposure=0.5, cpe="cpe:2.3:o:microsoft:windows_10:21h2:*"),
    AssetRecord(asset_id="ASSET-004", name="Core Router", asset_type="network_device",
                criticality=8.5, exposure=0.8, cpe="cpe:2.3:o:cisco:ios:15.6:*"),
    AssetRecord(asset_id="ASSET-005", name="Cloud Storage Bucket", asset_type="cloud",
                criticality=7.0, exposure=0.9, cpe=""),
    AssetRecord(asset_id="ASSET-006", name="Internal Wiki", asset_type="server",
                criticality=5.0, exposure=0.1, cpe="cpe:2.3:a:atlassian:confluence:7.13.0:*"),
    AssetRecord(asset_id="ASSET-007", name="VPN Gateway", asset_type="network_device",
                criticality=9.5, exposure=1.0, cpe="cpe:2.3:a:paloaltonetworks:pan-os:10.1:*"),
    AssetRecord(asset_id="ASSET-008", name="CI/CD Pipeline", asset_type="cloud",
                criticality=8.0, exposure=0.4, cpe=""),
]


def load_assets(path: str | Path | None = None) -> list[AssetRecord]:
    if path is None:
        logger.info("Using built-in sample assets (%d assets)", len(SAMPLE_ASSETS))
        return list(SAMPLE_ASSETS)

    p = Path(path)
    if not p.exists():
        logger.warning("Asset file %s not found — using sample assets", p)
        return list(SAMPLE_ASSETS)

    raw = json.loads(p.read_text(encoding="utf-8"))

    # Handle double-encoded JSON (string wrapping a JSON array)
    if isinstance(raw, str):
        raw = json.loads(raw)

    # Handle {"assets": [...]} wrapper
    if isinstance(raw, dict):
        raw = raw.get("assets") or raw.get("data") or list(raw.values())[0]

    if not isinstance(raw, list):
        logger.warning("Unexpected JSON structure in %s — using sample assets", p)
        return list(SAMPLE_ASSETS)

    assets = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            logger.warning("Skipping item %d in %s: expected object, got %s", i, p, type(item).__name__)
            continue
        try:
            assets.append(AssetRecord(**item))
        except Exception as exc:
            logger.warning("Skipping item %d in %s: %s", i, p, exc)

    if not assets:
        logger.warning("No valid assets parsed from %s — using sample assets", p)
        return list(SAMPLE_ASSETS)

    logger.info("Loaded %d assets from %s", len(assets), p)
    return assets
