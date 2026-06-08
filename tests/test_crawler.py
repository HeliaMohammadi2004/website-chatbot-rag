"""
tests/test_crawler.py
---------------------
Basic unit tests for the crawler module.
Run with: pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock
from src.crawler import _same_domain, _clean_text, crawl
from bs4 import BeautifulSoup


def test_same_domain_exact_match():
    assert _same_domain("https://example.com", "https://example.com/about") is True


def test_same_domain_different_domain():
    assert _same_domain("https://example.com", "https://other.com/page") is False


def test_same_domain_subdomain():
    assert _same_domain("https://example.com", "https://blog.example.com/post") is True


def test_clean_text_removes_scripts():
    html = """
    <html><body>
        <script>alert('hi')</script>
        <p>Hello world</p>
        <footer>Footer text</footer>
    </body></html>
    """
    soup = BeautifulSoup(html, "lxml")
    text = _clean_text(soup)
    assert "Hello world" in text
    assert "alert" not in text
    assert "Footer text" not in text


def test_clean_text_collapses_whitespace():
    html = "<html><body><p>   Hello   \n\n   World   </p></body></html>"
    soup = BeautifulSoup(html, "lxml")
    text = _clean_text(soup)
    assert "\n\n" not in text


@patch("src.crawler.requests.Session")
def test_crawl_returns_pages(mock_session_class):
    """Test that crawl returns a list of page dicts on successful fetch."""
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = """
    <html>
        <head><title>Test Page</title></head>
        <body><p>This is test content for the crawler.</p></body>
    </html>
    """
    mock_response.raise_for_status = MagicMock()
    mock_session.get.return_value = mock_response

    results = crawl("https://example.com", max_depth=0, max_pages=1, delay=0)
    assert len(results) == 1
    assert results[0]["title"] == "Test Page"
    assert "test content" in results[0]["text"]
    assert results[0]["url"] == "https://example.com"


@patch("src.crawler.requests.Session")
def test_crawl_skips_non_html(mock_session_class):
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session

    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "application/pdf"}
    mock_response.raise_for_status = MagicMock()
    mock_session.get.return_value = mock_response

    results = crawl("https://example.com/file.pdf", max_depth=0, max_pages=1, delay=0)
    assert results == []
