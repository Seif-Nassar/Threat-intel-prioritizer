from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class CVERecord(BaseModel):
    cve_id: str
    description: str
    cvss_score: float = 0.0
    cvss_vector: str = ""
    published: str = ""
    exploitability_score: float = 0.0
    impact_score: float = 0.0
    cwe_ids: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class AssetRecord(BaseModel):
    asset_id: str
    name: str
    asset_type: str  # server, workstation, network_device, cloud, etc.
    criticality: float  # 0.0 - 10.0
    exposure: float    # 0.0 - 1.0 (internet-facing = 1.0)
    cpe: str = ""      # CPE string for matching CVEs


class MITRETechnique(BaseModel):
    technique_id: str
    name: str
    tactic: str
    description: str
    detection: str = ""
    platforms: list[str] = Field(default_factory=list)


class RiskRecord(BaseModel):
    cve_id: str
    asset_id: str
    asset_name: str
    cvss_score: float
    exploitability_score: float
    asset_criticality: float
    asset_exposure: float
    mitre_techniques: list[str] = Field(default_factory=list)
    risk_score: float = 0.0          # ML-derived composite score
    priority: str = "LOW"            # CRITICAL / HIGH / MEDIUM / LOW
    recommendation: str = ""
