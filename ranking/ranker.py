"""
NeuroSearch - Hybrid Ranking Engine
===================================
This is what makes NeuroSearch different from a plain keyword search engine.

Three ranking strategies:

  1. KEYWORD (BM25)    - classic Information Retrieval. Scores a document by
                         how often the query words appear in it, adjusted for
                         document length and how rare each word is overall.
                         Great for exact matches. Fails when the user's words
                         don't literally appear in the document.

  2. SEMANTIC          - converts the query into a 384-dimensional vector using
                         a neural network, then finds documents whose vectors
                         point in a similar direction (cosine similarity).
                         Understands MEANING: "car" matches "automobile".

  3. HYBRID            - normalizes both scores to 0-1 and blends them:
                             final = (alpha * semantic) + ((1-alpha) * keyword)
                         Best of both worlds, and it's what we use by default.
"""

import os
import pickle

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_FILE = os.path.join(DATA_DIR, "index.pkl")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# How much weight semantic search gets in hybrid mode (0.0 - 1.0)
DEFAULT_ALPHA = 0.5


def normalize(scores):
    """Rescale an array of scores to the range 0-1 so they can be compared."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    lowest, highest = scores.min(), scores.max()
    if highest - lowest < 1e-9:
        return np.zeros_like(scores)
    return (scores - lowest) / (highest - lowest)


class NeuroRanker:
    """Loads the index once, then answers search queries."""

    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.bm25 = None
        self.model = None
        self.processor = None
        self.ready = False

    # ------------------------- LOADING -------------------------

    def load(self):
        if not os.path.exists(INDEX_FILE) or not os.path.exists(EMBEDDINGS_FILE):
            raise FileNotFoundError(
                "Index files missing. Run:\n"
                "  python crawler/crawler.py\n"
                "  python indexer/indexer.py"
            )

        print("Loading index...")
        with open(INDEX_FILE, "rb") as f:
            data = pickle.load(f)

        self.documents = data["documents"]

        # BM25 needs the tokenized corpus
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(data["bm25_corpus"])

        # Same tokenizer the indexer used, so queries are processed identically
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "indexer"))
        from indexer import TextProcessor
        self.processor = TextProcessor()

        print("Loading embeddings...")
        self.embeddings = np.load(EMBEDDINGS_FILE)

        print(f"Loading model '{EMBEDDING_MODEL}'...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        self.ready = True
        print(f"Ready. {len(self.documents)} documents indexed.\n")

    # ------------------------- SCORING -------------------------

    def keyword_scores(self, query):
        """BM25 score for every document."""
        tokens = self.processor.tokenize(query)
        if not tokens:
            return np.zeros(len(self.documents))
        return np.array(self.bm25.get_scores(tokens))

    def semantic_scores(self, query):
        """Cosine similarity between the query vector and every document vector."""
        query_vector = self.model.encode(query, normalize_embeddings=True)
        # Embeddings are normalized, so a dot product IS cosine similarity
        return self.embeddings @ query_vector

    # ------------------------- SEARCH -------------------------

    def search(self, query, mode="hybrid", alpha=DEFAULT_ALPHA, top_k=10):
        """
        Run a search and return ranked results.

        mode:  "keyword" | "semantic" | "hybrid"
        alpha: semantic weight in hybrid mode (0 = pure keyword, 1 = pure semantic)
        """
        if not self.ready:
            raise RuntimeError("Ranker not loaded. Call load() first.")

        query = query.strip()
        if not query:
            return []

        raw_keyword = self.keyword_scores(query)
        raw_semantic = self.semantic_scores(query)

        norm_keyword = normalize(raw_keyword)
        norm_semantic = normalize(raw_semantic)

        if mode == "keyword":
            final = norm_keyword
        elif mode == "semantic":
            final = norm_semantic
        else:  # hybrid
            final = (alpha * norm_semantic) + ((1 - alpha) * norm_keyword)

        # Get the indices of the top scoring documents, best first
        top_indices = np.argsort(final)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            if final[idx] <= 0.001:
                continue
            doc = self.documents[int(idx)]
            results.append({
                "rank": rank,
                "id": doc["id"],
                "title": doc["title"],
                "url": doc["url"],
                "snippet": doc["snippet"],
                "score": round(float(final[idx]), 4),
                # These two are what power the score breakdown bar in the UI —
                # showing WHY a result ranked where it did
                "keyword_score": round(float(norm_keyword[idx]), 4),
                "semantic_score": round(float(norm_semantic[idx]), 4),
            })
        return results

    def explain(self, query):
        """Return how the query was processed - useful for the demo/viva."""
        return {
            "original_query": query,
            "processed_tokens": self.processor.tokenize(query),
            "total_documents": len(self.documents),
            "embedding_dimensions": int(self.embeddings.shape[1]),
        }
