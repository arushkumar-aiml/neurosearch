
# 🔍 NeuroSearch

> An AI-powered search engine built from scratch — combining classical Information Retrieval with modern semantic search, so it understands *meaning*, not just keywords.

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/backend-Python-blue)
![JavaScript](https://img.shields.io/badge/frontend-JavaScript-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 What is NeuroSearch?

Most college search engine projects just do keyword matching. NeuroSearch goes further — it combines **traditional ranking algorithms (TF-IDF, BM25)** with **ML-based semantic search (sentence embeddings)** to return results based on meaning, not just exact word matches. It's a hybrid search engine, built entirely from scratch as a 15-day solo build.

## ✨ Features

- 🕷️ **Custom Web Crawler** — fetches and parses pages from scratch, respects `robots.txt`
- 📚 **Inverted Index** — the core search data structure, built without external search libraries
- 📊 **TF-IDF + BM25 Ranking** — classical relevance scoring
- 🧠 **Semantic Search** — sentence-transformer embeddings for meaning-based matching
- 🔀 **Hybrid Ranking Engine** — combines keyword + semantic scores for best-of-both results
- ⚡ **Fast JavaScript Frontend** — clean, responsive search UI with live results

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Crawler | Python, BeautifulSoup |
| NLP Preprocessing | NLTK / spaCy |
| Embeddings | sentence-transformers (Hugging Face) |
| Ranking | scikit-learn (TF-IDF), rank_bm25, cosine similarity |
| Backend API | Python, FastAPI |
| Frontend | JavaScript (Vanilla JS + Fetch API), HTML5, CSS3 |
| Data Storage | SQLite / JSON |

## 📁 Project Structure

```
neurosearch/
├── README.md
├── requirements.txt
├── .gitignore
├── crawler/
│   └── crawler.py
├── indexer/
│   └── indexer.py
├── ranking/
│   └── ranker.py
├── app/
│   └── main.py
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

## 🚀 Getting Started

```bash
git clone https://github.com/arushkumar-aiml/neurosearch.git
cd neurosearch
pip install -r requirements.txt
python app/main.py
```

Then open `frontend/index.html` in your browser.

## 🗺️ Build Roadmap (15 Days)

- [x] Day 1: Project setup, architecture, repo
- [ ] Day 2-3: Web crawler
- [ ] Day 4-5: Text preprocessing pipeline
- [ ] Day 6-7: Inverted index + TF-IDF search
- [ ] Day 8-9: BM25 ranking
- [ ] Day 10-11: Semantic search with embeddings
- [ ] Day 12: Hybrid ranking system
- [ ] Day 13: JavaScript frontend + backend integration
- [ ] Day 14: Testing & polish
- [ ] Day 15: Final demo & presentation

## 👨‍💻 Author

**Arush Kumar** — BTech CS-AIML  
GitHub: [@arushkumar-aiml](https://github.com/arushkumar-aiml)  
Building this in public, one day at a time. Follow the journey on LinkedIn.

## 📄 License

MIT
```

