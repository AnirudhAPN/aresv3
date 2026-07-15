"""
Resolves the Kaggle page a dataset recommendation should link to.

Two tiers, both real (never fabricated):
  1. A curated, HTTP-verified direct Kaggle dataset/competition page from
     the known-datasets whitelist (data/known_datasets.json), for datasets
     that have a confirmed live Kaggle mirror.
  2. A Kaggle search results page for the dataset's name — Kaggle's own
     search page always resolves, so this is a safe "most relevant Kaggle
     page" fallback for datasets without a direct mirror (e.g. meta-
     repositories like UCI/OpenML, or anything the LLM suggested that
     isn't on the whitelist at all).
"""
from __future__ import annotations

from urllib.parse import quote_plus


def _search_fallback(name: str) -> str:
    return f"https://www.kaggle.com/search?q={quote_plus(name)}"


def resolve_kaggle_url(name: str, known_datasets: list[dict]) -> str:
    normalized = name.strip().lower()
    for entry in known_datasets:
        if entry["name"].strip().lower() == normalized:
            kaggle_url = entry.get("kaggle_url")
            if kaggle_url:
                return kaggle_url
            break
    return _search_fallback(name)
