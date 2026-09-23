"""
NeuroSearch - API Server
========================
FastAPI backend that exposes the search engine over HTTP and serves the
JavaScript frontend.

Endpoints:
    GET  /                  -> the search UI
    GET  /api/search        -> run a search
    GET  /api/explain       -> show how a query gets processed
    GET  /api/stats         -> index statistics

Run:  python app/main.py
Then open http://localhost:8000
"""

import os
import sys
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Make the ranking module importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "ranking"))

from ranker import NeuroRanker  # noqa: E402

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = FastAPI(
    title="NeuroSearch API",
    description="Hybrid search engine - BM25 keyword search + semantic embeddings",
    version="1.0.0",
)

# Allow the frontend to call the API from anywhere (fine for a local project)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ranker = NeuroRanker()


@app.on_event("startup")
def startup():
    """Load the index once when the server boots, not on every request."""
    print("\n" + "=" * 55)
    print("  NeuroSearch starting up")
    print("=" * 55)
    try:
        ranker.load()
        print("=" * 55)
        print("  Server ready -> http://localhost:8000")
        print("=" * 55 + "\n")
    except FileNotFoundError as err:
        print(f"\n{err}\n")


# ------------------------- API ROUTES -------------------------

@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    mode: str = Query("hybrid", pattern="^(keyword|semantic|hybrid)$"),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Semantic weight"),
    limit: int = Query(10, ge=1, le=50),
):
    if not ranker.ready:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run crawler/crawler.py then indexer/indexer.py",
        )

    started = time.perf_counter()
    results = ranker.search(q, mode=mode, alpha=alpha, top_k=limit)
    elapsed_ms = (time.perf_counter() - started) * 1000

    return {
        "query": q,
        "mode": mode,
        "alpha": alpha,
        "count": len(results),
        "time_ms": round(elapsed_ms, 2),
        "results": results,
    }


@app.get("/api/suggest")
def suggest(prefix: str = Query(..., min_length=1)):
    """Autocomplete: vocabulary terms starting with `prefix`, most common first."""
    if not ranker.ready:
        return {"prefix": prefix, "suggestions": []}
    return {"prefix": prefix, "suggestions": ranker.suggest(prefix)}


@app.get("/api/topics")
def topics():
    """Every topic auto-discovered by KMeans clustering over the embeddings."""
    if not ranker.ready:
        return {"topics": []}
    return {"topics": ranker.get_topics()}


@app.get("/api/explain")
def explain(q: str = Query(..., min_length=1)):
    if not ranker.ready:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return ranker.explain(q)


@app.get("/api/stats")
def stats():
    if not ranker.ready:
        return {"ready": False, "documents": 0}
    return {
        "ready": True,
        "documents": len(ranker.documents),
        "embedding_dimensions": int(ranker.embeddings.shape[1]),
        "model": "all-MiniLM-L6-v2",
        "topics": len(ranker.cluster_labels),
    }


# ------------------------- FRONTEND -------------------------

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
