from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_builder import FEATURE_COLS
from src.models.risk_scorer import RiskScorer


def _synthetic_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
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
    return df[FEATURE_COLS]


def test_predict_range():
    scorer = RiskScorer()
    df = _synthetic_df()
    scorer.train(df)
    preds = scorer.predict(df)
    assert preds.min() >= 0.0
    assert preds.max() <= 1.0


def test_untrained_raises():
    scorer = RiskScorer()
    with pytest.raises(RuntimeError):
        scorer.predict(_synthetic_df(5))


def test_feature_importance_keys():
    scorer = RiskScorer()
    scorer.train(_synthetic_df())
    fi = scorer.feature_importance
    assert set(fi.keys()) == set(FEATURE_COLS)
