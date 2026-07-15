"""
Deterministic citation formatting for retrieved papers.

Pure code, no LLM involved — citations are built directly from the same
real, retrieved metadata (`Paper`) that the Literature Agent already
verified came from Semantic Scholar / arXiv, never generated or inferred.
"""
from __future__ import annotations

from models.schemas import Paper


def _apa_authors(authors: list[str]) -> str:
    """Format an author list APA-style: 'Last, F. M.' Falls back to the
    name as-is if it isn't in a splittable 'First Last' form."""

    def _one(name: str) -> str:
        parts = name.strip().split()
        if len(parts) < 2:
            return name.strip()
        last = parts[-1]
        initials = " ".join(f"{p[0]}." for p in parts[:-1] if p)
        return f"{last}, {initials}"

    formatted = [_one(a) for a in authors if a and a.strip()]
    if not formatted:
        return ""
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) <= 20:
        return ", ".join(formatted[:-1]) + ", & " + formatted[-1]
    # APA 7th: 19 authors, ellipsis, then the last author.
    return ", ".join(formatted[:19]) + ", ... " + formatted[-1]


def format_apa(paper: Paper) -> str:
    authors = _apa_authors(paper.authors)
    year = paper.year if paper.year is not None else "n.d."
    title = paper.title.strip().rstrip(".")
    venue = f" {paper.venue}." if paper.venue else ""
    pieces = []
    if authors:
        pieces.append(f"{authors}")
    pieces.append(f"({year}).")
    pieces.append(f"{title}.{venue}".strip())
    pieces.append(paper.url)
    return " ".join(p for p in pieces if p).strip()


def _bibtex_key(paper: Paper) -> str:
    first_author = paper.authors[0].split()[-1] if paper.authors and paper.authors[0].split() else "anon"
    year = paper.year or "nd"
    slug = "".join(ch for ch in paper.title if ch.isalnum())[:20]
    return f"{first_author}{year}{slug}".replace(" ", "")


def format_bibtex(paper: Paper) -> str:
    entry_type = "misc" if paper.source == "arxiv" else "article"
    key = _bibtex_key(paper)
    lines = [f"@{entry_type}{{{key},"]
    lines.append(f'  title = {{{paper.title}}},')
    if paper.authors:
        lines.append(f'  author = {{{" and ".join(paper.authors)}}},')
    if paper.year is not None:
        lines.append(f"  year = {{{paper.year}}},")
    if paper.venue:
        field = "journal" if entry_type == "article" else "note"
        lines.append(f"  {field} = {{{paper.venue}}},")
    lines.append(f"  url = {{{paper.url}}},")
    lines.append("}")
    return "\n".join(lines)


def annotate_citations(papers: list[Paper]) -> None:
    """Mutates each paper in place, filling citation_apa / citation_bibtex."""
    for paper in papers:
        paper.citation_apa = format_apa(paper)
        paper.citation_bibtex = format_bibtex(paper)
