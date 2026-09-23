# 🔍 NeuroSearch

> An AI-powered search engine built from scratch — combining classical Information Retrieval with modern semantic search, so it understands *meaning*, not just keywords.

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/backend-Python-blue)
![JavaScript](https://img.shields.io/badge/frontend-JavaScript-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 What is NeuroSearch?

Most college search engine projects just do keyword matching. NeuroSearch goes further — it combines **traditional ranking algorithms (TF-IDF, BM25)** with **ML-based semantic search (sentence embeddings)** to return results based on meaning, not just exact word matches.

The interface exposes the ranking itself: a live slider blends keyword and semantic scoring, and every result shows a colour-coded breakdown of *why* it ranked where it did.

## ✨ Features

- 🕷️ **Custom Web Crawler** — BFS crawl, respects `robots.txt`, extracts clean article text
- 📚 **Inverted Index** — the core search data structure, built from scratch
- 📊 **TF-IDF + BM25 Ranking** — classical relevance scoring
- 🧠 **Semantic Search** — sentence-transformer embeddings for meaning-based matching
- 🔀 **Hybrid Ranking Engine** — adjustable blend of keyword + semantic scores
- 🔬 **Explainable Results** — per-result score breakdown showing each signal's contribution
- 🗂️ **AI Topic Clustering** — KMeans over the embeddings auto-tags every result with a topic, no manual labels
- ✂️ **ML-Generated Snippets** — shows the sentence closest in *meaning* to your query, not a static substring
- 🔎 **Smart Autocomplete** — vocabulary-ranked query suggestions as you type
- ⚡ **JavaScript Frontend** — responsive search UI with live re-ranking, red/black themed

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Crawler | Python, BeautifulSoup, requests |
| NLP Preprocessing | NLTK (tokenize, stopwords, Porter stemming) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Ranking | rank_bm25, scikit-learn TF-IDF, cosine similarity |
| Backend API | Python, FastAPI, Uvicorn |
| Frontend | JavaScript (Vanilla JS + Fetch API), HTML5, CSS3 |
| Data Storage | JSON + pickle + NumPy arrays |

## 📁 Project Structure

```
neurosearch/
├── README.md
├── SETUP.md
├── requirements.txt
├── .gitignore
├── crawler/
│   └── crawler.py          # BFS web crawler
├── indexer/
│   └── indexer.py          # preprocessing, inverted index, TF-IDF, embeddings
├── ranking/
│   └── ranker.py           # BM25 + semantic + hybrid ranking
├── app/
│   └── main.py             # FastAPI server
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── data/                   # generated files (gitignored)
```

## 🚀 Getting Started

```bash
git clone https://github.com/arushkumar-aiml/neurosearch.git
cd neurosearch

pip install -r requirements.txt

python crawler/crawler.py     # step 1: collect pages (~5 min)
python indexer/indexer.py     # step 2: build the index (~3 min)
python app/main.py            # step 3: start the server
```

Open **http://localhost:8000**

Full instructions and troubleshooting are in [SETUP.md](SETUP.md).

## 🔌 API

| Endpoint | Description |
|---|---|
| `GET /api/search?q=...&mode=hybrid&alpha=0.5` | Run a search |
| `GET /api/explain?q=...` | Show how a query is tokenized |
| `GET /api/stats` | Index statistics |
| `GET /api/suggest?prefix=...` | Autocomplete suggestions |
| `GET /api/topics` | AI-discovered topic clusters |

## 🧠 How the Ranking Works

```
query ──┬─→ tokenize ──→ BM25 ─────────→ keyword score ──┐
        │                                                ├─→ normalize → blend → rank
        └─→ embed ────→ cosine similarity → semantic ────┘
                                              score
```

`final = (alpha × semantic) + ((1 − alpha) × keyword)`

Set `alpha = 0` for pure keyword search, `alpha = 1` for pure semantic, anything between for hybrid.

## 🎤 Presentation / Viva Demo

A scripted terminal walkthrough for presenting the project (problem, architecture, USPs, tech stack, then a live keyword-vs-semantic-vs-hybrid comparison on a query you type in):

```bash
python presentation.py
```

## 🗺️ Build Roadmap (15 Days)

- [x] Day 1: Project setup, architecture, repo
- [x] Day 2-3: Web crawler
- [x] Day 4-5: Text preprocessing pipeline
- [x] Day 6-7: Inverted index + TF-IDF search
- [x] Day 8-9: BM25 ranking
- [x] Day 10-11: Semantic search with embeddings
- [x] Day 12: Hybrid ranking system
- [x] Day 13: JavaScript frontend + backend integration
- [ ] Day 14: Testing & polish
- [ ] Day 15: Final demo & presentation

## 👨‍💻 Author

**Arush Kumar** — BTech CS-AIML  
GitHub: [@arushkumar-aiml](https://github.com/arushkumar-aiml)

## 📄 License

MIT
