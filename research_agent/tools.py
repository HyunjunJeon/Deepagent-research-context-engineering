"""리서치 도구 모듈.

이 모듈은 리서치 에이전트를 위한 검색 및 콘텐츠 처리 유틸리티를 제공하며,
Tavily 를 사용해 URL 을 찾고 전체 웹페이지 콘텐츠를 가져와 마크다운으로 변환한다.
"""

from typing import Annotated, Literal

import httpx
from dotenv import load_dotenv
from langchain_core.tools import InjectedToolArg, tool
from markdownify import markdownify
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient()


def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """웹페이지 콘텐츠를 가져와 마크다운으로 변환한다.

    Args:
        url: 가져올 URL
        timeout: 요청 타임아웃 (초 단위)

    Returns:
        마크다운 형식의 웹페이지 콘텐츠
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return markdownify(response.text)
    except Exception as e:
        return f"Error fetching content from {url}: {str(e)}"


@tool()
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """Search the web for information on a given query.

    Uses Tavily to discover relevant URLs, then fetches and returns full webpage content as markdown.

    Args:
        query: Search query to execute
        max_results: Maximum number of results to return (default: 1)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

    Returns:
        Formatted search results with full webpage content
    """
    # Tavily 를 사용해 관련 URL 목록을 조회한다
    search_results = tavily_client.search(
        query,
        max_results=max_results,
        topic=topic,
    )

    # 각 URL 에 대해 전체 콘텐츠를 가져온다
    result_texts = []
    for result in search_results.get("results", []):
        url = result["url"]
        title = result["title"]

        # 웹페이지 콘텐츠를 가져온다
        content = fetch_webpage_content(url)

        result_text = f"""## {title}
**URL:** {url}

{content}

---
"""
        result_texts.append(result_text)

    # 최종 응답 형식으로 정리한다
    response = f"""Found {len(result_texts)} result(s) for '{query}':

{chr(10).join(result_texts)}"""

    return response


@tool()
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"
