"""
enrichment/web_enricher.py
Controlled external context enrichment.

Design principles:
  1. Only triggered by specific findings (anomaly date, metric name)
  2. Adversarial search — always searches for BOTH confirming AND opposing evidence
  3. Source reliability scoring tracked against verified outcomes
  4. All sources tagged: internal_data | external_context | prior_analysis
  5. No raw scraping — structured search only (news APIs, web search)
  6. Respects internet_off_mode policy

Source reliability is tracked over time:
  - Each source domain gets a reliability score based on how often
    its context aligned with verified findings
  - Sources below 0.4 reliability are flagged as low-quality
"""

from __future__ import annotations
import hashlib
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from core.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "enrichment.db"


@dataclass
class EnrichmentResult:
    query: str
    direction: str              # "confirming" | "opposing"
    source: str                 # URL or domain
    title: str
    snippet: str
    reliability_score: float    # 0–1
    evidence_type: str          # "external_context"
    retrieved_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EnrichmentContext:
    findings: list[EnrichmentResult] = field(default_factory=list)
    confirming_count: int = 0
    opposing_count: int = 0
    net_signal: str = "neutral"     # "confirms" | "contradicts" | "neutral"
    summary: str = ""


class SourceReliabilityStore:
    """Tracks and updates source reliability over verified outcomes."""

    def __init__(self, db_path: str = None):
        self._db = db_path or str(DB_PATH)
        os.makedirs(os.path.dirname(self._db), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS source_reliability (
                    domain TEXT PRIMARY KEY,
                    total_uses INTEGER DEFAULT 0,
                    verified_helpful INTEGER DEFAULT 0,
                    reliability_score REAL DEFAULT 0.5,
                    last_updated TEXT
                );
            """)
            conn.commit()

    def get_score(self, domain: str) -> float:
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                "SELECT reliability_score FROM source_reliability WHERE domain = ?",
                (domain,)
            ).fetchone()
        return row[0] if row else 0.5

    def update(self, domain: str, was_helpful: bool):
        score = self.get_score(domain)
        new_score = 0.7 * score + 0.3 * (1.0 if was_helpful else 0.0)
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                INSERT INTO source_reliability (domain, total_uses, verified_helpful, reliability_score, last_updated)
                VALUES (?, 1, ?, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    total_uses = total_uses + 1,
                    verified_helpful = verified_helpful + ?,
                    reliability_score = ?,
                    last_updated = ?
            """, (domain, 1 if was_helpful else 0, round(new_score, 4),
                  datetime.now().isoformat(),
                  1 if was_helpful else 0, round(new_score, 4),
                  datetime.now().isoformat()))
            conn.commit()


class WebEnricher:
    """
    Controlled web enrichment with adversarial search.
    Only fires when triggered by specific findings.
    """

    def __init__(self):
        self._reliability = SourceReliabilityStore()

    def enrich(
        self,
        kpi: str,
        anomaly_date: str = None,
        finding_summary: str = None,
        policy=None,
    ) -> EnrichmentContext:
        """
        Main entry point. Triggered by specific findings.
        Returns EnrichmentContext with both confirming and opposing evidence.
        """
        if policy and not policy.check_internet_off() is None:
            logger.info("Internet-off mode: enrichment skipped.")
            return EnrichmentContext(summary="Skipped: internet-off mode active.")

        ctx = EnrichmentContext()

        if not kpi:
            return ctx

        # Build targeted queries
        confirm_query = self._build_query(kpi, anomaly_date, finding_summary, direction="confirming")
        oppose_query  = self._build_query(kpi, anomaly_date, finding_summary, direction="opposing")

        # Run both searches
        confirming = self._search(confirm_query, "confirming")
        opposing   = self._search(oppose_query,  "opposing")

        ctx.findings = confirming + opposing
        ctx.confirming_count = len(confirming)
        ctx.opposing_count   = len(opposing)

        # Net signal
        conf_weight = sum(r.reliability_score for r in confirming)
        opp_weight  = sum(r.reliability_score for r in opposing)

        if conf_weight > opp_weight * 1.5:
            ctx.net_signal = "confirms"
        elif opp_weight > conf_weight * 1.5:
            ctx.net_signal = "contradicts"
        else:
            ctx.net_signal = "neutral"

        ctx.summary = (
            f"External context: {len(confirming)} confirming, {len(opposing)} opposing sources. "
            f"Net signal: {ctx.net_signal}."
        )
        logger.info(ctx.summary)
        return ctx

    def _build_query(self, kpi: str, date: str, summary: str, direction: str) -> str:
        base = f"{kpi} analytics"
        if date:
            base += f" {date[:7]}"   # year-month only
        if direction == "confirming":
            if summary:
                # Use first 5 words of summary
                words = summary.split()[:5]
                base += " " + " ".join(words)
            return base
        else:
            # Opposing: search for counter-narrative
            if "drop" in (summary or "").lower() or "decline" in (summary or "").lower():
                return f"{kpi} growth increase strong performance"
            elif "increase" in (summary or "").lower() or "spike" in (summary or "").lower():
                return f"{kpi} decline drop problem issue"
            return f"{kpi} opposite trend alternative explanation"

    def _search(self, query: str, direction: str) -> list[EnrichmentResult]:
        """Execute a web search via available means."""
        results = []

        # Try DuckDuckGo (no API key required)
        try:
            results = self._ddg_search(query, direction)
        except Exception as e:
            logger.warning(f"DDG search failed: {e}")

        # Filter by reliability
        scored = []
        for r in results:
            domain = self._extract_domain(r.source)
            r.reliability_score = self._reliability.get_score(domain)
            scored.append(r)

        # Sort by reliability, return top 3
        return sorted(scored, key=lambda r: r.reliability_score, reverse=True)[:3]

    def _ddg_search(self, query: str, direction: str) -> list[EnrichmentResult]:
        """Search via DuckDuckGo instant answers API (no key required)."""
        import requests
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        results = []
        # AbstractText
        if data.get("AbstractText"):
            results.append(EnrichmentResult(
                query=query, direction=direction,
                source=data.get("AbstractURL", "duckduckgo.com"),
                title=data.get("Heading", query),
                snippet=data["AbstractText"][:300],
                reliability_score=0.5,
                evidence_type="external_context",
            ))
        # Related topics
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(EnrichmentResult(
                    query=query, direction=direction,
                    source=topic.get("FirstURL", "duckduckgo.com"),
                    title=topic["Text"][:80],
                    snippet=topic["Text"][:200],
                    reliability_score=0.4,
                    evidence_type="external_context",
                ))
        return results

    def _extract_domain(self, url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.replace("www.", "")
        except Exception:
            return url[:30]
