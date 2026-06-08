"""
crawler.py
----------
Crawls a website starting from a given URL.
Stays within the same domain, respects max_depth and max_pages limits.
Returns a list of dicts: {url, title, text}
"""

import re
import time
import logging
from urllib.parse import urljoin, urlparse
from typing import List, Dict

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; WebChatBot/1.0; +https://github.com/your-repo)"
    )
}


def _same_domain(base_url: str, url: str) -> bool:
    """Return True if `url` belongs to the same domain as `base_url`."""
    base_netloc = urlparse(base_url).netloc
    url_netloc = urlparse(url).netloc
    # Allow sub-domains of the same root
    return url_netloc == base_netloc or url_netloc.endswith("." + base_netloc)


def _clean_text(soup: BeautifulSoup) -> str:
    """Extract clean visible text from a BeautifulSoup object."""
    # Remove script, style, nav, footer, header tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def crawl(
    start_url: str,
    max_depth: int = 3,
    max_pages: int = 50,
    delay: float = 0.5,
) -> List[Dict[str, str]]:
    """
    Crawl a website starting from `start_url`.

    Parameters
    ----------
    start_url : str
        The entry point URL.
    max_depth : int
        Maximum link-follow depth (default 3).
    max_pages : int
        Maximum number of pages to crawl (default 50).
    delay : float
        Seconds to wait between requests (politeness).

    Returns
    -------
    List[Dict]
        Each dict has keys: 'url', 'title', 'text'
    """
    visited: set[str] = set()
    results: List[Dict[str, str]] = []
    # Queue: (url, depth)
    queue: List[tuple[str, int]] = [(start_url, 0)]

    session = requests.Session()
    session.headers.update(HEADERS)

    while queue and len(results) < max_pages:
        url, depth = queue.pop(0)

        # Normalise URL — strip fragments
        url = url.split("#")[0].rstrip("/")
        if not url:
            continue

        if url in visited:
            continue
        visited.add(url)

        try:
            response = session.get(url, timeout=10, allow_redirects=True)
            # Only process HTML pages
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            continue

        soup = BeautifulSoup(response.text, "lxml")

        title = soup.title.get_text(strip=True) if soup.title else url
        text = _clean_text(soup)

        if text:
            results.append({"url": url, "title": title, "text": text})
            logger.info("[%d/%d] Crawled: %s", len(results), max_pages, url)

        # Discover links if not at max depth
        if depth < max_depth:
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"].strip()
                # Skip anchors, mailto, tel, javascript
                if href.startswith(("#", "mailto:", "tel:", "javascript:")):
                    continue
                full_url = urljoin(url, href).split("#")[0].rstrip("/")
                if full_url not in visited and _same_domain(start_url, full_url):
                    queue.append((full_url, depth + 1))

        time.sleep(delay)

    logger.info("Crawl finished. Total pages: %d", len(results))
    return results
