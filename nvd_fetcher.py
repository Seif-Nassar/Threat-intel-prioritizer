from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.schema import AssetRecord, CVERecord, MITRETechnique

if TYPE_CHECKING:
    from src.features.vulberta_embedder import VulBertaEmbedder

logger = logging.getLogger(__name__)

_HIGH_RISK_CWES = {"CWE-78", "CWE-79", "CWE-89", "CWE-119", "CWE-120", "CWE-22", "CWE-434"}

TABULAR_FEATURE_COLS = [
    "cvss_score",
    "exploitability",
    "impact_score",
    "asset_criticality",
    "asset_exposure",
    "exposure_x_exploitability",
    "criticality_x_impact",
    "high_risk_cwe",
    "mitre_coverage",
]

FEATURE_COLS = TABULAR_FEATURE_COLS


def _precompute_technique_words(techniques: list[MITRETechnique]) -> list[set[str]]:
    """Compute word sets for all techniques once — reused across all CVEs."""
    return [
        set(re.findall(r"\b\w{4,}\b", t.description.lower()))
        for t in techniques
    ]


def _mitre_coverage_fast(desc_words: set[str], tech_word_sets: list[set[str]]) -> float:
    if not tech_word_sets:
        return 0.0
    matches = sum(1 for tws in tech_word_sets if desc_words & tws)
    return min(1.0, matches / len(tech_word_sets) * 50)


def build_features(
    cves: list[CVERecord],
    assets: list[AssetRecord],
    techniques: list[MITRETechnique],
    embedder: VulBertaEmbedder | None = None,
) -> pd.DataFrame:
    # Pre-compute technique word sets once (not per-CVE)
    tech_word_sets = _precompute_technique_words(techniques)
    logger.info(
        "Building features: %d CVEs × %d assets (MITRE: %d techniques)",
        len(cves), len(assets), len(techniques),
    )

    rows = []
    log_every = max(1, len(cves) // 10)  # log progress every 10%

    for i, cve in enumerate(cves):
        if i % log_every == 0:
            logger.info("  Feature build progress: %d / %d CVEs", i, len(cves))

        desc_words = set(re.findall(r"\b\w{4,}\b", cve.description.lower()))
        mitre_cov = _mitre_coverage_fast(desc_words, tech_word_sets)
        high_risk_cwe = int(any(c in _HIGH_RISK_CWES for c in cve.cwe_ids))

        for asset in assets:
            rows.append({
                "cve_id": cve.cve_id,
                "asset_id": asset.asset_id,
                "asset_name": asset.name,
                "cvss_score": cve.cvss_score,
                "exploitability": cve.exploitability_score,
                "impact_score": cve.impact_score,
                "asset_criticality": asset.criticality,
                "asset_exposure": asset.exposure,
                "exposure_x_exploitability": asset.exposure * cve.exploitability_score,
                "criticality_x_impact": asset.criticality * cve.impact_score,
                "high_risk_cwe": high_risk_cwe,
                "mitre_coverage": mitre_cov,
                "_desc": cve.description,
            })

    logger.info("Feature rows built: %d", len(rows))
    df = pd.DataFrame(rows)

    if embedder is not None:
        unique_cves = df[["cve_id", "_desc"]].drop_duplicates("cve_id")
        n_unique = len(unique_cves)
        logger.info("Encoding %d unique CVE descriptions with CodeBERT…", n_unique)
        embs = embedder.encode(unique_cves["_desc"].tolist())
        emb_df = pd.DataFrame(embs, columns=embedder.col_names)
        emb_df["cve_id"] = unique_cves["cve_id"].values
        df = df.merge(emb_df, on="cve_id", how="left")
        logger.info("CodeBERT embeddings attached (%d dims)", embedder.dim)

    df = df.drop(columns=["_desc"])
    logger.info("Feature matrix ready: %s", df.shape)
    return df


def get_model_feature_cols(df: pd.DataFrame) -> list[str]:
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    return TABULAR_FEATURE_COLS + emb_cols
