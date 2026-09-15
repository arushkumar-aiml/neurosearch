# NeuroSearch — Setup Guide

Complete instructions to get the project running on Windows, macOS or Linux.

---

## 1. Requirements

- **Python 3.9 or newer** — check with `python --version`
- ~2 GB free disk space (the ML model downloads once, ~90 MB)
- Internet connection for the crawl and the first model download

---

## 2. Install dependencies

Open a terminal inside the `neurosearch` folder.

**Create a virtual environment (recommended):**

Windows:
```cmd
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Install packages:**
```bash
pip install -r requirements.txt
```

This takes a few minutes — `sentence-transformers` pulls in PyTorch, which is large.

---

## 3. Run the pipeline

The three steps must run in this order. Each one produces files the next one needs.

### Step 1 — Crawl (~5 minutes)
```bash
python crawler/crawler.py
```
Crawls 300 Wikipedia pages starting from AI/ML topics and saves them to `data/documents.json`.

To crawl a different topic, edit `SEED_URLS` at the top of `crawler/crawler.py`.  
To crawl more or fewer pages, change `MAX_PAGES`.

### Step 2 — Build the index (~3 minutes)
```bash
python indexer/indexer.py
```
Tokenizes and stems all text, builds the inverted index and TF-IDF matrix, then generates semantic embeddings. The first run downloads the `all-MiniLM-L6-v2` model.

Produces `data/index.pkl` and `data/embeddings.npy`.

### Step 3 — Start the server
```bash
python app/main.py
```

Open **http://localhost:8000** in your browser.

---

## 4. Using the interface

- **Keyword mode** — pure BM25. Matches exact words. Try `neural network`.
- **Semantic mode** — pure embeddings. Try `how do computers understand what people write` — no exact keyword overlap, but relevant results still come back. This is the demo moment.
- **Hybrid mode** — blends both. Drag the slider and watch results re-rank live.

Each result shows a two-colour bar: orange is the keyword contribution, teal is the semantic contribution.

---

## 5. Troubleshooting

**`ModuleNotFoundError: No module named 'sentence_transformers'`**  
Dependencies didn't install. Re-run `pip install -r requirements.txt`. If you made a venv, make sure it's activated.

**`Index files missing`**  
Run the crawler and indexer first (steps 1 and 2 above), then restart the server.

**NLTK stopwords error**  
Run once in Python:
```python
import nltk
nltk.download("stopwords")
```

**Crawler returns very few pages**  
Wikipedia may be rate-limiting. Increase `CRAWL_DELAY` in `crawler/crawler.py` to `1.0` and try again.

**Port 8000 already in use**  
Change the port at the bottom of `app/main.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8080)
```

**Model download is slow**  
It's a one-time ~90 MB download, cached afterwards in `~/.cache/huggingface`.

---

## 6. Push to GitHub

```bash
git init
git add .
git commit -m "NeuroSearch: hybrid search engine from scratch"
git branch -M main
git remote add origin https://github.com/arushkumar-aiml/neurosearch.git
git push -u origin main
```

`data/` is gitignored — the index is rebuilt locally, not committed.

---

## 7. Things worth understanding before your viva

Your examiner will likely ask about these. Read the comments in each file:

| Question | Where to look |
|---|---|
| What is an inverted index and why use one? | `indexer/indexer.py` → `InvertedIndex` class |
| How does BM25 differ from TF-IDF? | `ranking/ranker.py` → module docstring |
| What is an embedding? | `indexer/indexer.py` → step 5 |
| Why normalize scores before blending? | `ranking/ranker.py` → `normalize()` |
| How does the crawler avoid infinite loops? | `crawler/crawler.py` → `visited` set + BFS queue |
