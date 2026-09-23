"""
NeuroSearch - Presentation / Viva Demo Script
==============================================
A scripted, terminal-based walkthrough of the project for a college
presentation or viva. Explains the problem, the architecture, the
unique selling points, and then runs a LIVE side-by-side comparison
of Keyword vs Semantic vs Hybrid search on a real query typed in by
whoever is presenting.

Works with or without the `rich` library:
  - If `rich` is installed, you get coloured panels/tables.
  - If not, it falls back to plain formatted text — nothing breaks.

Run:  python presentation.py
"""

import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "ranking"))
sys.path.insert(0, os.path.join(BASE_DIR, "indexer"))

# ------------------------- OPTIONAL RICH UI -------------------------

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None


def pause(seconds=1.2):
    time.sleep(seconds)


def heading(title):
    if RICH:
        console.rule(f"[bold red]{title}[/bold red]", style="red")
    else:
        print("\n" + "=" * 70)
        print(title.upper())
        print("=" * 70)


def bullet(text):
    if RICH:
        console.print(f"  [red]-[/red] {text}")
    else:
        print(f"  - {text}")


def banner():
    text = r"""
 _   _                       ____                      _
| \ | | ___ _   _ _ __ ___  / ___|  ___  __ _ _ __ ___| |__
|  \| |/ _ \ | | | '__/ _ \ \___ \ / _ \/ _` | '__/ __| '_ \
| |\  |  __/ |_| | | | (_) | ___) |  __/ (_| | | | (__| | | |
|_| \_|\___|\__,_|_|  \___/ |____/ \___|\__,_|_|  \___|_| |_|
"""
    if RICH:
        console.print(Text(text, style="bold red"))
        console.print(
            Panel.fit(
                "AI-powered search — combines classical IR with neural semantic search",
                border_style="red",
            )
        )
    else:
        print(text)
        print("AI-powered search — combines classical IR with neural semantic search")


# ------------------------- SLIDES -------------------------

def slide_problem():
    heading("The Problem")
    bullet("Most student search-engine projects only do keyword / substring matching")
    bullet('They fail the moment a user\'s words differ from the document\'s words')
    bullet('Example: searching "car" won\'t find a page that only says "automobile"')
    pause()


def slide_solution():
    heading("The Idea: Hybrid Search")
    bullet("Keyword layer (BM25) — classical Information Retrieval, exact-term precision")
    bullet("Semantic layer (sentence embeddings) — a neural net that understands MEANING")
    bullet("A live slider blends the two: final = alpha*semantic + (1-alpha)*keyword")
    pause()


def slide_architecture():
    heading("Architecture")
    stages = [
        ("1. Crawler", "BFS crawl, robots.txt-respecting, extracts clean article text"),
        ("2. Indexer", "Tokenize -> inverted index -> TF-IDF -> sentence embeddings -> topic clusters"),
        ("3. Ranker", "BM25 + cosine similarity, blended, with ML-generated snippets"),
        ("4. API", "FastAPI server exposing /api/search, /api/suggest, /api/topics"),
        ("5. Frontend", "Vanilla JS UI with live re-ranking, autocomplete, topic badges"),
    ]
    if RICH:
        table = Table(box=box.SIMPLE_HEAVY, border_style="red", header_style="bold red")
        table.add_column("Stage")
        table.add_column("What it does")
        for name, desc in stages:
            table.add_row(name, desc)
        console.print(table)
    else:
        for name, desc in stages:
            print(f"  {name:<14} {desc}")
    pause()


def slide_usp():
    heading("Unique Selling Points")
    bullet("Explainable ranking — every result shows WHY it ranked there (keyword vs semantic split), not a black box")
    bullet("AI topic clustering — KMeans over the embeddings auto-tags every result with a topic, zero manual labels")
    bullet("ML-generated snippets — the shown snippet is the sentence whose MEANING is closest to the query, not a static substring")
    bullet("Built from scratch — inverted index, ranking blend and clustering are hand-implemented, not just a wrapped library call")
    pause()


def slide_tech_stack():
    heading("Tech Stack")
    rows = [
        ("Crawler", "Python, BeautifulSoup, requests"),
        ("NLP preprocessing", "NLTK — tokenize, stopwords, stemming"),
        ("Keyword ranking", "rank_bm25, scikit-learn TF-IDF"),
        ("Semantic ranking", "sentence-transformers (all-MiniLM-L6-v2)"),
        ("Topic discovery", "scikit-learn KMeans over embeddings"),
        ("Backend", "FastAPI + Uvicorn"),
        ("Frontend", "Vanilla JavaScript, HTML5, CSS3"),
    ]
    if RICH:
        table = Table(box=box.SIMPLE_HEAVY, border_style="red", header_style="bold red")
        table.add_column("Layer")
        table.add_column("Technology")
        for a, b in rows:
            table.add_row(a, b)
        console.print(table)
    else:
        for a, b in rows:
            print(f"  {a:<20} {b}")
    pause()


# ------------------------- LIVE DEMO -------------------------

def slide_live_demo():
    heading("Live Demo")
    try:
        from ranker import NeuroRanker
    except Exception as err:
        bullet(f"Could not import the ranker: {err}")
        return

    ranker = NeuroRanker()
    try:
        ranker.load()
    except FileNotFoundError:
        bullet("Index not built yet on this machine.")
        bullet("Run: python crawler/crawler.py   then   python indexer/indexer.py")
        return

    if RICH:
        query = console.input("\n[bold red]Type a query to demo live -> [/bold red]").strip()
    else:
        query = input("\nType a query to demo live -> ").strip()

    if not query:
        bullet("No query entered, skipping live demo.")
        return

    for mode in ("keyword", "semantic", "hybrid"):
        heading(f'Mode: {mode.upper()}   |   query: "{query}"')
        results = ranker.search(query, mode=mode, alpha=0.5, top_k=3)
        if not results:
            bullet("No results.")
            continue
        for r in results:
            topic = f"  [{r['topic']}]" if r.get("topic") else ""
            if RICH:
                console.print(f"  [bold]{r['rank']}. {r['title']}[/bold]{topic}  "
                               f"[red]score={r['score']}[/red]")
                console.print(f"     {r['snippet'][:140]}")
            else:
                print(f"  {r['rank']}. {r['title']}{topic}  score={r['score']}")
                print(f"     {r['snippet'][:140]}")
        pause(0.6)

    heading("What just happened")
    bullet("Keyword mode only matched literal words from your query")
    bullet("Semantic mode matched by meaning, even with different wording")
    bullet("Hybrid mode blended both — this is NeuroSearch's default")


def slide_roadmap():
    heading("Roadmap / Status")
    bullet("Day 1-13: crawler, index, BM25, semantic search, hybrid ranking, frontend — DONE")
    bullet("Added: topic clustering, ML snippets, autocomplete, red/black themed UI")
    bullet("Day 14-15: testing, polish, final demo")


def closing():
    heading("Thank You")
    bullet("NeuroSearch — Arush Kumar, BTech CS-AIML")
    bullet("github.com/arushkumar-aiml/neurosearch")


# ------------------------- MAIN -------------------------

def main():
    banner()
    pause(0.8)
    slide_problem()
    slide_solution()
    slide_architecture()
    slide_usp()
    slide_tech_stack()
    slide_live_demo()
    slide_roadmap()
    closing()


if __name__ == "__main__":
    main()
