"""
NeuroSearch - Indexer
=====================
Takes the crawled documents and builds everything the search engine needs:

  1. Text preprocessing  - tokenize, lowercase, remove stopwords, stem
  2. Inverted index      - {word: [doc_ids]}  <- the core search data structure
  3. TF-IDF matrix       - classical keyword relevance scores
  4. Semantic embeddings - vector representation of meaning (the ML part)

Everything gets saved to data/ so the API can load it instantly at startup.

Run:  python indexer/indexer.py
"""

import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np

# ------------------------- CONFIGURATION -------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DOCUMENTS_FILE = os.path.join(DATA_DIR, "documents.json")
INDEX_FILE = os.path.join(DATA_DIR, "index.pkl")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")

# Small, fast sentence-transformer model. 384-dimensional vectors.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# How much text from each document to embed (model has a token limit)
EMBED_CHAR_LIMIT = 1000


# ------------------------- TEXT PREPROCESSING -------------------------

class TextProcessor:
    """
    Turns raw text into clean tokens.

    Pipeline:  lowercase -> strip punctuation -> tokenize
               -> remove stopwords -> stem (running/ran/runs -> run)
    """

    def __init__(self):
        self.stemmer = None
        self.stopwords = self._load_stopwords()
        self._setup_stemmer()

    def _setup_stemmer(self):
        try:
            from nltk.stem import PorterStemmer
            self.stemmer = PorterStemmer()
        except ImportError:
            print("  [warn] NLTK not available, skipping stemming")

    def _load_stopwords(self):
        """Load NLTK stopwords, with a built-in fallback list."""
        try:
            import nltk
            from nltk.corpus import stopwords
            try:
                words = set(stopwords.words("english"))
            except LookupError:
                print("  Downloading NLTK stopwords (one time)...")
                nltk.download("stopwords", quiet=True)
                words = set(stopwords.words("english"))
            return words
        except ImportError:
            return {
                "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
                "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
                "to", "was", "were", "will", "with", "this", "these", "those",
                "but", "or", "not", "can", "which", "their", "they", "have",
            }

    def tokenize(self, text):
        """Convert a string into a list of clean, stemmed tokens."""
        text = text.lower()
        # Keep only letters, numbers and spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()

        clean = []
        for token in tokens:
            if len(token) < 2 or token in self.stopwords:
                continue
            if self.stemmer:
                token = self.stemmer.stem(token)
            clean.append(token)
        return clean


# ------------------------- INVERTED INDEX -------------------------

class InvertedIndex:
    """
    The heart of any search engine.

    Instead of scanning every document for a word (slow), we store a map:
        "machine" -> {0: 5, 12: 2, 47: 9}    (doc_id -> how many times it appears)

    So looking up a word is instant, no matter how many documents there are.
    """

    def __init__(self):
        self.index = defaultdict(dict)   # word -> {doc_id: term_frequency}
        self.doc_lengths = {}            # doc_id -> number of tokens
        self.total_docs = 0

    def add_document(self, doc_id, tokens):
        self.doc_lengths[doc_id] = len(tokens)
        self.total_docs += 1
        for token in tokens:
            self.index[token][doc_id] = self.index[token].get(doc_id, 0) + 1

    def get_documents(self, term):
        """Return {doc_id: frequency} for a term."""
        return self.index.get(term, {})

    def vocabulary_size(self):
        return len(self.index)

    def average_doc_length(self):
        if not self.doc_lengths:
            return 0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)


# ------------------------- BUILD PIPELINE -------------------------

def build_index():
    # ---- Step 1: load crawled documents ----
    if not os.path.exists(DOCUMENTS_FILE):
        print("ERROR: data/documents.json not found.")
        print("Run the crawler first:  python crawler/crawler.py")
        return

    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} documents\n")

    # ---- Step 2: preprocess + build inverted index ----
    print("Building inverted index...")
    processor = TextProcessor()
    inverted_index = InvertedIndex()
    tokenized_docs = []

    for doc in documents:
        # Title counted twice so title matches rank higher
        full_text = f"{doc['title']} {doc['title']} {doc['text']}"
        tokens = processor.tokenize(full_text)
        tokenized_docs.append(tokens)
        inverted_index.add_document(doc["id"], tokens)

    print(f"  Vocabulary size: {inverted_index.vocabulary_size():,} unique terms")
    print(f"  Average doc length: {inverted_index.average_doc_length():.0f} tokens\n")

    # ---- Step 3: BM25 corpus (uses the same tokens) ----
    print("Preparing BM25 corpus...")
    bm25_corpus = tokenized_docs

    # ---- Step 4: TF-IDF matrix ----
    print("Building TF-IDF matrix...")
    from sklearn.feature_extraction.text import TfidfVectorizer

    joined_docs = [" ".join(tokens) for tokens in tokenized_docs]
    tfidf_vectorizer = TfidfVectorizer(max_features=50000, sublinear_tf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(joined_docs)
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}\n")

    # ---- Step 5: semantic embeddings (the ML/AIML part) ----
    print(f"Generating semantic embeddings with '{EMBEDDING_MODEL}'...")
    print("  (first run downloads the model, ~90MB)")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts_to_embed = [
        f"{doc['title']}. {doc['text'][:EMBED_CHAR_LIMIT]}" for doc in documents
    ]
    embeddings = model.encode(
        texts_to_embed,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # lets us use dot product as cosine similarity
    )
    print(f"  Embeddings shape: {embeddings.shape}\n")

    # ---- Step 6: topic clustering (the second ML layer) ----
    # Unsupervised KMeans over the semantic embeddings -> every document gets
    # auto-tagged with a topic, with ZERO manual labeling. Cluster labels are
    # derived from the cluster's own top TF-IDF terms, not hand-written.
    print("Discovering topics (KMeans over embeddings)...")
    from sklearn.cluster import KMeans

    n_clusters = max(1, min(10, len(documents) // 8 or 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    doc_cluster = {}
    cluster_labels = {}
    for cluster_id in range(n_clusters):
        member_idx = [i for i, c in enumerate(cluster_ids) if c == cluster_id]
        for i in member_idx:
            doc_cluster[documents[i]["id"]] = int(cluster_id)

        if member_idx:
            # Average TF-IDF vector across the cluster's members, then take
            # the highest-weighted terms as a human-readable label.
            avg_tfidf = np.asarray(tfidf_matrix[member_idx].mean(axis=0)).ravel()
            top_term_idx = avg_tfidf.argsort()[::-1][:3]
            label = " / ".join(feature_names[i].title() for i in top_term_idx)
        else:
            label = f"Topic {cluster_id + 1}"
        cluster_labels[cluster_id] = label

    print(f"  {n_clusters} topics discovered:")
    for cid, label in cluster_labels.items():
        count = sum(1 for c in cluster_ids if c == cid)
        print(f"    [{cid}] {label}  ({count} docs)")
    print()

    # ---- Step 7: save everything ----
    print("Saving index files...")
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(
            {
                "documents": documents,
                "inverted_index": dict(inverted_index.index),
                "doc_lengths": inverted_index.doc_lengths,
                "bm25_corpus": bm25_corpus,
                "tfidf_vectorizer": tfidf_vectorizer,
                "tfidf_matrix": tfidf_matrix,
                "doc_cluster": doc_cluster,
                "cluster_labels": cluster_labels,
            },
            f,
        )

    np.save(EMBEDDINGS_FILE, embeddings)

    print(f"  {os.path.abspath(INDEX_FILE)}")
    print(f"  {os.path.abspath(EMBEDDINGS_FILE)}")
    print("\nIndex built successfully. Now run:  python app/main.py")


if __name__ == "__main__":
    build_index()
