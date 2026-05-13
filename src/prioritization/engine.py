from __future__ import annotations

import logging

import pandas as pd

from src.features.vulberta_embedder import VulBertaEmbedder
from src.ingestion.asset_loader import load_assets
from src.ingestion.mitre_loader import load_techniques
from src.ingestion.nvd_fetcher import fetch_recent_cves, load_cves_from_nvd_feed
from src.models.risk_scorer import RiskScorer
from src.features.feature_builder import build_features
from src.schema import RiskRecord

logger = logging.getLogger(__name__)

_PRIORITY_THRESHOLDS = [
    (0.75, "CRITICAL"),
    (0.50, "HIGH"),
    (0.30, "MEDIUM"),
    (0.0,  "LOW"),
]

_RECOMMENDATIONS: dict[str, str] = {
    "CRITICAL": "Patch immediately. Isolate affected asset if patch unavailable.",
    "HIGH":     "Schedule patch within 24-72 hours. Apply compensating controls.",
    "MEDIUM":   "Include in next patch cycle (within 30 days). Monitor for PoC.",
    "LOW":      "Track and patch in routine maintenance window.",
}


def _priority(score: float) -> str:
    for threshold, label in _PRIORITY_THRESHOLDS:
        if score >= threshold:
            return label
    return "LOW"


def run_pipeline(
    days_back: int = 30,
    keywords: list[str] | None = None,
    asset_path: str | None = None,
    cve_feed_path: str | None = None,
    max_cves: int = 500,
    load_mitre: bool = True,
    embedder: VulBertaEmbedder | None = None,
) -> tuple[list[RiskRecord], RiskScorer, dict]:
    # 1. Ingest
    if cve_feed_path:
        logger.info("Loading CVEs from uploaded feed: %s", cve_feed_path)
        cves = load_cves_from_nvd_feed(cve_feed_path)
    else:
        cves = fetch_recent_cves(days_back=days_back, keywords=keywords)

    if len(cves) > max_cves:
        logger.info("Capping CVEs: %d → %d (max_cves limit)", len(cves), max_cves)
        cves = cves[:max_cves]

    assets = load_assets(asset_path)
    techniques = load_techniques() if load_mitre else []

    if not cves:
        logger.warning("No CVEs fetched — returning empty results")
        return [], RiskScorer(), {}

    # 2. Build features (VulBERTa embeddings attached when embedder is provided)
    df = build_features(cves, assets, techniques, embedder=embedder)

    # 3. Train / score — invalidate cached model when embedding dims change
    scorer = RiskScorer()
    loaded = scorer.load()
    if loaded:
        expected_cols = set(scorer._feature_cols)
        actual_cols = set(col for col in df.columns if col not in
                         ("cve_id", "asset_id", "asset_name", "tactic"))
        if expected_cols != actual_cols:
            logger.info("Feature set changed — retraining model.")
            loaded = False

    if not loaded:
        scorer.train(df)
        scorer.save()

    df["risk_score"] = scorer.predict(df)
    df["priority"] = df["risk_score"].apply(_priority)

    # 4. Build output records
    records: list[RiskRecord] = []
    for _, row in df.iterrows():
        records.append(RiskRecord(
            cve_id=row["cve_id"],
            asset_id=row["asset_id"],
            asset_name=row["asset_name"],
            cvss_score=row["cvss_score"],
            exploitability_score=row["exploitability"],
            asset_criticality=row["asset_criticality"],
            asset_exposure=row["asset_exposure"],
            risk_score=round(float(row["risk_score"]), 4),
            priority=row["priority"],
            recommendation=_RECOMMENDATIONS[row["priority"]],
        ))

    records.sort(key=lambda r: r.risk_score, reverse=True)

    stats = {
        "total_cves": len(cves),
        "total_assets": len(assets),
        "total_pairs": len(records),
        "critical": sum(1 for r in records if r.priority == "CRITICAL"),
        "high": sum(1 for r in records if r.priority == "HIGH"),
        "medium": sum(1 for r in records if r.priority == "MEDIUM"),
        "low": sum(1 for r in records if r.priority == "LOW"),
        "feature_importance": scorer.feature_importance,
        "vulberta_enabled": embedder is not None,
    }
    logger.info("Pipeline complete. Stats: %s", stats)
    return records, scorer, stats
