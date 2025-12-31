#!/usr/bin/env python3
"""arXiv Academic Paper Search.

Searches the arXiv preprint repository for research papers in physics,
mathematics, computer science, quantitative biology, and related fields.

Usage:
    python arxiv_search.py "query" [--max-papers N] [--output-format FORMAT]

Examples:
    python arxiv_search.py "transformer attention mechanism"
    python arxiv_search.py "deep learning drug discovery" --max-papers 5
    python arxiv_search.py "large language model" --output-format json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def query_arxiv(
    query: str,
    max_papers: int = 10,
    output_format: str = "text",
) -> str:
    """Query arXiv for papers based on the provided search query.

    Parameters
    ----------
    query : str
        The search query string. Supports arXiv query syntax including:
        - Simple keywords: "neural network"
        - Author search: "author:Hinton"
        - Category filter: "cat:cs.LG"
        - Boolean operators: "transformer AND attention"
    max_papers : int
        The maximum number of papers to retrieve (default: 10).
    output_format : str
        Output format: "text", "json", or "markdown" (default: "text").

    Returns:
    -------
    str
        The formatted search results or an error message.
    """
    try:
        import arxiv
    except ImportError:
        return (
            "Error: arxiv package not installed.\n"
            "Install with: pip install arxiv\n"
            "Or if using a virtual environment: .venv/bin/python -m pip install arxiv"
        )

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_papers,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: list[dict[str, Any]] = []
        for paper in client.results(search):
            papers.append({
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown",
                "arxiv_id": paper.entry_id.split("/")[-1] if paper.entry_id else "Unknown",
                "url": paper.entry_id or "",
                "summary": paper.summary.replace("\n", " ").strip(),
                "categories": paper.categories,
                "pdf_url": paper.pdf_url or "",
            })

        if not papers:
            return "No papers found on arXiv matching your query."

        return format_output(papers, query, output_format)

    except Exception as e:
        return f"Error querying arXiv: {e}"


def format_output(papers: list[dict[str, Any]], query: str, output_format: str) -> str:
    """Format the search results based on the specified output format.

    Parameters
    ----------
    papers : list[dict[str, Any]]
        List of paper dictionaries.
    query : str
        The original search query.
    output_format : str
        Output format: "text", "json", or "markdown".

    Returns
    -------
    str
        Formatted output string.
    """
    if output_format == "json":
        return json.dumps(
            {
                "query": query,
                "total_results": len(papers),
                "papers": papers,
            },
            indent=2,
            ensure_ascii=False,
        )

    elif output_format == "markdown":
        lines = [f"# arXiv Search Results: {query}\n"]
        lines.append(f"**Total Results:** {len(papers)}\n")

        for paper in papers:
            lines.append(f"## {paper['title']}\n")
            lines.append(f"**Authors:** {', '.join(paper['authors'])}")
            lines.append(f"**Published:** {paper['published']}")
            lines.append(f"**arXiv ID:** [{paper['arxiv_id']}]({paper['url']})")
            lines.append(f"**Categories:** {', '.join(paper['categories'])}")
            lines.append(f"**PDF:** [{paper['arxiv_id']}.pdf]({paper['pdf_url']})\n")
            lines.append("### Abstract\n")
            lines.append(f"{paper['summary']}\n")
            lines.append("---\n")

        return "\n".join(lines)

    else:  # text format (default)
        lines = []
        lines.append(f"arXiv Search Results for: {query}")
        lines.append(f"Total Results: {len(papers)}")
        lines.append("=" * 80)

        for paper in papers:
            lines.append(f"\nTitle: {paper['title']}")
            lines.append(f"Authors: {', '.join(paper['authors'])}")
            lines.append(f"Published: {paper['published']}")
            lines.append(f"arXiv ID: {paper['arxiv_id']}")
            lines.append(f"URL: {paper['url']}")
            lines.append(f"Categories: {', '.join(paper['categories'])}")
            lines.append(f"PDF: {paper['pdf_url']}")
            lines.append("-" * 80)
            lines.append(f"Summary: {paper['summary']}")
            lines.append("=" * 80)

        return "\n".join(lines)


def main() -> None:
    """Main entry point for the arXiv search CLI."""
    parser = argparse.ArgumentParser(
        description="Search arXiv for academic research papers",
        epilog="""
Examples:
  %(prog)s "transformer attention mechanism"
  %(prog)s "deep learning" --max-papers 5
  %(prog)s "cat:cs.LG neural network" --output-format json
  %(prog)s "author:Hinton representation learning" --output-format markdown

Query Syntax:
  - Simple keywords: "neural network pruning"
  - Author search: "author:Bengio"
  - Category filter: "cat:cs.CV object detection"
  - Boolean: "transformer AND self-attention"
  - Exact phrase: '"attention is all you need"'
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "query",
        type=str,
        help="Search query string (supports arXiv query syntax)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=10,
        help="Maximum number of papers to retrieve (default: 10)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format: text, json, or markdown (default: text)",
    )

    args = parser.parse_args()

    result = query_arxiv(
        query=args.query,
        max_papers=args.max_papers,
        output_format=args.output_format,
    )
    print(result)


if __name__ == "__main__":
    main()
