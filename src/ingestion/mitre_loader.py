from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

from src.schema import MITRETechnique

logger = logging.getLogger(__name__)

MITRE_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
)
CACHE_PATH = Path("data/mitre_enterprise.json")


def _download_stix(url: str, cache: Path) -> dict:
    if cache.exists():
        logger.info("Loading MITRE ATT&CK from cache: %s", cache)
        return json.loads(cache.read_text(encoding="utf-8"))
    logger.info("Downloading MITRE ATT&CK STIX bundle…")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data


def load_techniques(url: str = MITRE_URL, cache: Path = CACHE_PATH) -> list[MITRETechnique]:
    bundle = _download_stix(url, cache)
    objects = bundle.get("objects", [])

    techniques: list[MITRETechnique] = []
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        ext_refs = obj.get("external_references", [])
        tid = next(
            (r["external_id"] for r in ext_refs if r.get("source_name") == "mitre-attack"),
            "",
        )
        if not tid:
            continue

        tactic = ""
        phases = obj.get("kill_chain_phases", [])
        if phases:
            tactic = phases[0].get("phase_name", "")

        detection = obj.get("x_mitre_detection", "")
        platforms = obj.get("x_mitre_platforms", [])
        desc = obj.get("description", "")

        techniques.append(MITRETechnique(
            technique_id=tid,
            name=obj.get("name", ""),
            tactic=tactic,
            description=desc[:400],
            detection=detection[:400],
            platforms=platforms,
        ))

    logger.info("Loaded %d MITRE ATT&CK techniques", len(techniques))
    return techniques
