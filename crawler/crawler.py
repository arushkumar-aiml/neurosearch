"""
NeuroSearch - Web Crawler
=========================
Crawls web pages starting from seed URLs using Breadth-First Search (BFS),
extracts clean text, and saves everything to data/documents.json.

How it works:
  1. Put seed URLs in a queue.
  2. Pop a URL, check robots.txt permission, download the page.
  3. Extract the title + main body text (strip scripts, navs, footers).
  4. Find all links on the page, add new ones to the queue.
  5. Repeat until MAX_PAGES is reached.

Run:  python crawler/crawler.py
"""

import json
import os
import time
from collections import deque
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# ------------------------- CONFIGURATION -------------------------

# Seed URLs - crawl starts from here. Change these to crawl a different topic.
SEED_URLS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Information_retrieval",
    "https://en.wikipedia.org/wiki/Search_engine",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
]

# Only crawl URLs on this domain (keeps the crawl focused and legal)
ALLOWED_DOMAIN = "en.wikipedia.org"

MAX_PAGES = 300          # How many pages to crawl in total
CRAWL_DELAY = 0.5        # Seconds to wait between requests (be polite!)
MIN_TEXT_LENGTH = 300    # Skip pages with less text than this
REQUEST_TIMEOUT = 10     # Seconds before giving up on a request

USER_AGENT = "NeuroSearchBot/1.0 (Educational college project)"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "documents.json")

# URL patterns to skip (Wikipedia meta/admin pages, files, etc.)
SKIP_PATTERNS = [
    ":", "#", "Main_Page", "index.php", ".jpg", ".png", ".svg", ".pdf",
]


# ------------------------- ROBOTS.TXT -------------------------

class RobotsChecker:
    """Reads robots.txt for each domain and caches the result."""

    def __init__(self, user_agent):
        self.user_agent = user_agent
        self.cache = {}

    def can_fetch(self, url):
        domain = urlparse(url).netloc
        if domain not in self.cache:
            parser = RobotFileParser()
            parser.set_url(f"https://{domain}/robots.txt")
            try:
                parser.read()
                self.cache[domain] = parser
            except Exception:
                # If robots.txt can't be read, be conservative and allow
                self.cache[domain] = None
        parser = self.cache[domain]
        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)


# ------------------------- HELPERS -------------------------

def is_valid_url(url):
    """Check whether a URL should be crawled."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.netloc != ALLOWED_DOMAIN:
        return False
    if not parsed.path.startswith("/wiki/"):
        return False
    for pattern in SKIP_PATTERNS:
        if pattern in parsed.path:
            return False
    return True


def clean_url(url):
    """Remove #fragments and ?query strings so we don't crawl duplicates."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def extract_content(html, url):
    """Pull the title and readable body text out of raw HTML."""
    soup = BeautifulSoup(html, "lxml")

    # Remove everything that isn't real content
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "noscript", "table"]):
        tag.decompose()

    # Title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else urlparse(url).path.split("/")[-1]

    # Body: Wikipedia keeps article text inside this div
    body = soup.find("div", {"class": "mw-parser-output"}) or soup.find("body")
    if body is None:
        return None

    paragraphs = [p.get_text(" ", strip=True) for p in body.find_all("p")]
    text = " ".join(para for para in paragraphs if len(para) > 40)

    if len(text) < MIN_TEXT_LENGTH:
        return None

    return {
        "url": url,
        "title": title,
        "text": text,
        # First ~250 chars used as the snippet shown in search results
        "snippet": text[:250].rsplit(" ", 1)[0] + "...",
    }


def extract_links(html, base_url):
    """Find all crawlable links on a page."""
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for anchor in soup.find_all("a", href=True):
        full_url = clean_url(urljoin(base_url, anchor["href"]))
        if is_valid_url(full_url):
            links.add(full_url)
    return links


# ------------------------- MAIN CRAWLER -------------------------

def crawl():
    robots = RobotsChecker(USER_AGENT)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    queue = deque(SEED_URLS)
    visited = set()
    documents = []

    print(f"Starting crawl -> target: {MAX_PAGES} pages\n")

    while queue and len(documents) < MAX_PAGES:
        url = clean_url(queue.popleft())

        if url in visited:
            continue
        visited.add(url)

        if not robots.can_fetch(url):
            print(f"  [robots.txt] skipped: {url}")
            continue

        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                continue

            doc = extract_content(response.text, url)
            if doc:
                doc["id"] = len(documents)
                documents.append(doc)
                print(f"  [{len(documents):>3}/{MAX_PAGES}] {doc['title'][:60]}")

            # Add newly found links to the back of the queue (BFS)
            for link in extract_links(response.text, url):
                if link not in visited:
                    queue.append(link)

        except requests.RequestException as err:
            print(f"  [error] {url} -> {err}")

        time.sleep(CRAWL_DELAY)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Crawled {len(documents)} pages.")
    print(f"Saved to: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    crawl()
