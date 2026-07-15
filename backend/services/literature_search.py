"""
Retrieval-first literature search: Semantic Scholar (primary) + arXiv
(supplement, esp. for CS/AI/ML topics). Never asks an LLM to invent paper
metadata — every `Paper` returned here comes from a real API response.
"""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

import httpx

from config import settings
from models.schemas import Paper
from services.cache import literature_cache

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


class LiteratureSearchService:
    def __init__(self) -> None:
        self.timeout = 20.0

    # -- Semantic Scholar ------------------------------------------------

    def _search_semantic_scholar(self, query: str, limit: int = 10, max_retries: int = 2) -> list[Paper]:
        headers = {}
        if settings.semantic_scholar_api_key:
            headers["x-api-key"] = settings.semantic_scholar_api_key

        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,authors,year,venue,citationCount,url,externalIds",
        }
        url = f"{settings.semantic_scholar_base_url}/paper/search"
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            for attempt in range(max_retries + 1):
                resp = client.get(url, params=params, headers=headers)
                if resp.status_code == 429 and attempt < max_retries:
                    wait = 2 ** attempt * 3  # 3s, 6s — the shared free-tier limit resets fast
                    logging.warning(
                        "Semantic Scholar rate-limited (attempt %d/%d). Retrying in %ds...",
                        attempt + 1, max_retries + 1, wait,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break

        papers: list[Paper] = []
        for item in data.get("data", []):
            if not item.get("abstract"):
                continue  # skip papers we can't synthesize from
            papers.append(
                Paper(
                    title=item.get("title", "Untitled"),
                    authors=[a.get("name", "") for a in item.get("authors", [])],
                    year=item.get("year"),
                    venue=item.get("venue") or None,
                    url=item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
                    citation_count=item.get("citationCount"),
                    source="semantic_scholar",
                    abstract=item.get("abstract", ""),
                    summary="",
                    verified=True,
                )
            )
        return papers

    # -- arXiv -------------------------------------------------------------

    def _search_arxiv(self, query: str, limit: int = 10) -> list[Paper]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            resp = client.get(settings.arxiv_base_url, params=params)
            resp.raise_for_status()
            xml_text = resp.text

        root = ET.fromstring(xml_text)
        papers: list[Paper] = []
        for entry in root.findall("atom:entry", ARXIV_NS):
            title_el = entry.find("atom:title", ARXIV_NS)
            summary_el = entry.find("atom:summary", ARXIV_NS)
            id_el = entry.find("atom:id", ARXIV_NS)
            published_el = entry.find("atom:published", ARXIV_NS)
            authors = [
                a.findtext("atom:name", default="", namespaces=ARXIV_NS)
                for a in entry.findall("atom:author", ARXIV_NS)
            ]
            if title_el is None or summary_el is None or id_el is None:
                continue
            year = None
            if published_el is not None and published_el.text:
                year = int(published_el.text[:4])
            papers.append(
                Paper(
                    title=" ".join(title_el.text.split()) if title_el.text else "Untitled",
                    authors=authors,
                    year=year,
                    venue="arXiv preprint",
                    url=id_el.text.strip() if id_el.text else "",
                    citation_count=None,
                    source="arxiv",
                    abstract=" ".join(summary_el.text.split()) if summary_el.text else "",
                    summary="",
                    verified=True,
                )
            )
        return papers

    # -- Merge / dedupe / rank ---------------------------------------------

    def _deduplicate(self, papers: list[Paper]) -> list[Paper]:
        unique: list[Paper] = []
        for paper in papers:
            if any(_title_similarity(paper.title, u.title) > 0.85 for u in unique):
                continue
            unique.append(paper)
        return unique

    def _rank(self, papers: list[Paper]) -> list[Paper]:
        def score(p: Paper) -> float:
            citation_score = (p.citation_count or 0)
            recency_score = (p.year or 0)
            return citation_score * 0.6 + recency_score * 0.4

        return sorted(papers, key=score, reverse=True)

    def search(self, keywords: list[str]) -> tuple[list[Paper], bool]:
        """Returns (papers, low_coverage). Cached per keyword-set (30-day TTL)."""
        cached = literature_cache.get(keywords)
        if cached is not None:
            papers = [Paper(**p) for p in cached["papers"]]
            return papers, cached["low_coverage"]

        query = " ".join(keywords[:6]) if keywords else ""
        all_papers: list[Paper] = []
        try:
            all_papers.extend(self._search_semantic_scholar(query, limit=10))
        except httpx.HTTPError as exc:
            logging.warning("Semantic Scholar search failed, falling back to arXiv-only: %s", exc)
        try:
            all_papers.extend(self._search_arxiv(query, limit=10))
        except httpx.HTTPError as exc:
            logging.warning("arXiv search failed: %s", exc)

        deduped = self._deduplicate(all_papers)
        ranked = self._rank(deduped)
        top = ranked[: settings.max_papers_returned]
        low_coverage = len(top) < settings.min_papers_for_full_coverage

        literature_cache.set(
            keywords,
            {"papers": [p.model_dump() for p in top], "low_coverage": low_coverage},
        )
        return top, low_coverage


literature_search_service = LiteratureSearchService()
