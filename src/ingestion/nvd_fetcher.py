from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import nvdlib

from src.schema import CVERecord

logger = logging.getLogger(__name__)


def load_cves_from_nvd_feed(path: str | Path) -> list[CVERecord]:
    """Parse an NVD JSON feed file (format 2.0) downloaded from nvd.nist.gov."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))

    vulnerabilities = raw.get("vulnerabilities", [])
    logger.info("Parsing NVD feed: %d entries in %s", len(vulnerabilities), p.name)

    records: list[CVERecord] = []
    for item in vulnerabilities:
        cve = item.get("cve", {})
        try:
            cve_id = cve.get("id", "")

            # Description (English)
            desc = ""
            for d in cve.get("descriptions", []):
                if d.get("lang") == "en":
                    desc = d.get("value", "")
                    break

            # CVSS scores — prefer v3.1 > v3.0 > v2
            metrics = cve.get("metrics", {})
            cvss = 0.0
            vector = ""
            exploitability = 0.0
            impact = 0.0

            for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                entries = metrics.get(key, [])
                if not entries:
                    continue
                m = entries[0]
                data = m.get("cvssData", {})
                cvss = float(data.get("baseScore", 0) or 0)
                vector = data.get("vectorString", "")
                exploitability = float(m.get("exploitabilityScore", 0) or 0)
                impact = float(m.get("impactScore", 0) or 0)
                break

            # CWEs
            cwes = []
            for w in cve.get("weaknesses", []):
                for d in w.get("description", []):
                    val = d.get("value", "")
                    if val.startswith("CWE-"):
                        cwes.append(val)

            # References
            refs = [r["url"] for r in cve.get("references", [])[:5] if "url" in r]

            records.append(CVERecord(
                cve_id=cve_id,
                description=desc,
                cvss_score=cvss,
                cvss_vector=vector,
                published=cve.get("published", ""),
                exploitability_score=exploitability,
                impact_score=impact,
                cwe_ids=cwes,
                references=refs,
            ))
        except Exception as exc:
            logger.debug("Skipping entry %s: %s", cve.get("id", "?"), exc)

    logger.info("Loaded %d CVE records from feed", len(records))
    return records


def fetch_recent_cves(days_back: int = 30, keywords: list[str] | None = None) -> list[CVERecord]:
    """Fetch CVEs published in the last `days_back` days from NVD.

    When keywords are given, each keyword is fetched separately and results
    are merged — the NVD API AND-matches all terms in a single call, which
    returns very few results when multiple unrelated keywords are combined.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)
    pub_start = start.strftime("%Y-%m-%dT%H:%M:%S.000")
    pub_end = end.strftime("%Y-%m-%dT%H:%M:%S.000")

    logger.info("Fetching CVEs from NVD: %s → %s", pub_start, pub_end)

    api_key = os.environ.get("NVD_API_KEY") or None
    delay = 0.2 if api_key else 0.6  # 6 req/sec with key, ~1.5 req/sec without

    # Build one search per keyword (or one search with no keyword filter)
    search_terms: list[str | None] = keywords if keywords else [None]
    per_term_limit = max(50, 200 // len(search_terms))

    seen: set[str] = set()
    all_results: list = []

    for term in search_terms:
        kwargs: dict = {"pubStartDate": pub_start, "pubEndDate": pub_end, "limit": per_term_limit}
        if term:
            kwargs["keywordSearch"] = term
        if api_key:
            kwargs["key"] = api_key
        try:
            results = nvdlib.searchCVE(**kwargs)
            for r in results:
                if r.id not in seen:
                    seen.add(r.id)
                    all_results.append(r)
            logger.info("Keyword '%s': %d results", term or "<none>", len(results))
        except Exception as exc:
            logger.warning("NVD fetch failed for keyword '%s': %s", term, exc)
        time.sleep(delay)

    records: list[CVERecord] = []
    for r in all_results:
        try:
            cvss = 0.0
            vector = ""
            exploitability = 0.0
            impact = 0.0
            if hasattr(r, "v31score") and r.v31score:
                cvss = float(r.v31score)
                vector = getattr(r, "v31vector", "")
                exploitability = float(getattr(r, "v31exploitabilityScore", 0) or 0)
                impact = float(getattr(r, "v31impactScore", 0) or 0)
            elif hasattr(r, "v30score") and r.v30score:
                cvss = float(r.v30score)
                vector = getattr(r, "v30vector", "")
                exploitability = float(getattr(r, "v30exploitabilityScore", 0) or 0)
                impact = float(getattr(r, "v30impactScore", 0) or 0)
            elif hasattr(r, "v2score") and r.v2score:
                cvss = float(r.v2score)
                exploitability = float(getattr(r, "v2exploitabilityScore", 0) or 0)
                impact = float(getattr(r, "v2impactScore", 0) or 0)

            desc = ""
            if r.descriptions:
                for d in r.descriptions:
                    if d.lang == "en":
                        desc = d.value
                        break

            cwes = []
            if hasattr(r, "cwe") and r.cwe:
                cwes = [c.value for c in r.cwe if hasattr(c, "value")]

            refs = []
            if hasattr(r, "references") and r.references:
                refs = [ref.url for ref in r.references[:5] if hasattr(ref, "url")]

            records.append(CVERecord(
                cve_id=r.id,
                description=desc,
                cvss_score=cvss,
                cvss_vector=vector,
                published=str(getattr(r, "published", "")),
                exploitability_score=exploitability,
                impact_score=impact,
                cwe_ids=cwes,
                references=refs,
            ))
        except Exception as exc:
            logger.debug("Skipping CVE %s: %s", getattr(r, "id", "?"), exc)

        time.sleep(0.05)  # stay within NVD rate limit

    logger.info("Fetched %d CVE records total (%d unique NVD results)", len(records), len(all_results))
    return records
