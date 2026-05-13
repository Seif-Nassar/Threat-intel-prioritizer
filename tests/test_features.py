from __future__ import annotations

import pandas as pd
import pytest

from src.features.feature_builder import build_features, FEATURE_COLS
from src.ingestion.asset_loader import SAMPLE_ASSETS
from src.schema import CVERecord


@pytest.fixture
def sample_cves():
    return [
        CVERecord(cve_id="CVE-2024-0001", description="Remote code execution via buffer overflow",
                  cvss_score=9.8, exploitability_score=3.9, impact_score=5.9,
                  cwe_ids=["CWE-119"]),
        CVERecord(cve_id="CVE-2024-0002", description="SQL injection in login form",
                  cvss_score=7.5, exploitability_score=2.8, impact_score=4.2,
                  cwe_ids=["CWE-89"]),
        CVERecord(cve_id="CVE-2024-0003", description="Information disclosure in API endpoint",
                  cvss_score=4.3, exploitability_score=2.0, impact_score=1.4,
                  cwe_ids=["CWE-200"]),
    ]


def test_build_features_shape(sample_cves):
    df = build_features(sample_cves, SAMPLE_ASSETS, [])
    assert len(df) == len(sample_cves) * len(SAMPLE_ASSETS)
    for col in FEATURE_COLS:
        assert col in df.columns


def test_high_risk_cwe_flag(sample_cves):
    df = build_features(sample_cves, SAMPLE_ASSETS, [])
    rce_rows = df[df["cve_id"] == "CVE-2024-0001"]
    assert rce_rows["high_risk_cwe"].all()
    info_rows = df[df["cve_id"] == "CVE-2024-0003"]
    assert not info_rows["high_risk_cwe"].any()


def test_interaction_features(sample_cves):
    df = build_features(sample_cves, SAMPLE_ASSETS, [])
    expected = df["asset_exposure"] * df["exploitability"]
    pd.testing.assert_series_equal(
        df["exposure_x_exploitability"].round(6),
        expected.round(6),
        check_names=False,
    )
