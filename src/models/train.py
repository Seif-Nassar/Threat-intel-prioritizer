"""Standalone training script — generates synthetic data and saves the model.

Usage:
    python src/models/train.py              # tabular features only
    python src/models/train.py --vulberta   # include VulBERTa embeddings (needs network)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from src.features.feature_builder import TABULAR_FEATURE_COLS
from src.models.risk_scorer import RiskScorer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_SAMPLE_DESCRIPTIONS = [
    "Remote code execution vulnerability in Apache HTTP Server allows unauthenticated attackers to execute arbitrary code.",
    "SQL injection in login form enables privilege escalation to database administrator.",
    "Buffer overflow in network stack leads to denial of service or arbitrary code execution.",
    "Information disclosure via path traversal exposes sensitive configuration files.",
    "Cross-site scripting allows injection of malicious scripts into web pages.",
    "Authentication bypass in VPN gateway permits unauthorized network access.",
    "Command injection in administrative interface enables full system compromise.",
    "Memory corruption in PDF parser leads to arbitrary code execution.",
]


def _generate_synthetic(n: int = 2000, with_embeddings: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "cvss_score":              rng.uniform(0, 10, n),
        "exploitability":          rng.uniform(0, 10, n),
        "impact_score":            rng.uniform(0, 10, n),
        "asset_criticality":       rng.uniform(0, 10, n),
        "asset_exposure":          rng.uniform(0, 1,  n),
        "high_risk_cwe":           rng.integers(0, 2, n),
        "mitre_coverage":          rng.uniform(0, 1,  n),
    })
    df["exposure_x_exploitability"] = df["asset_exposure"] * df["exploitability"]
    df["criticality_x_impact"]      = df["asset_criticality"] * df["impact_score"]
    df = df[TABULAR_FEATURE_COLS]

    if with_embeddings:
        from src.features.vulberta_embedder import VulBertaEmbedder
        embedder = VulBertaEmbedder()
        texts = [_SAMPLE_DESCRIPTIONS[i % len(_SAMPLE_DESCRIPTIONS)] for i in range(n)]
        embs = embedder.encode(texts)
        emb_df = pd.DataFrame(embs, columns=embedder.col_names)
        df = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vulberta", action="store_true", help="Include VulBERTa embeddings")
    args = parser.parse_args()

    df = _generate_synthetic(with_embeddings=args.vulberta)
    scorer = RiskScorer()
    scorer.train(df)
    scorer.save()
    mode = "with VulBERTa" if args.vulberta else "tabular only"
    print(f"Model trained ({mode}) and saved.")
