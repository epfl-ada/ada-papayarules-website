import argparse
import hashlib
import json
import os
import re
from typing import Tuple, List, Iterable
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors


"""USAGE:
$ python src/models/knn.py "trump" --k 10 --debug

Finds top-K subreddits similar to a query based on a learned text→embedding projection.
"""


# I/O & preprocessing 

def load_subreddit_embeddings(path: str) -> Tuple[pd.Index, np.ndarray]:
    """
    Load subreddit embeddings from CSV:
      - headerless: col0=subreddit, col1..=dims
      - headered: first column is subreddit name, remaining are dims
    Returns (names Index, L2-normalized float32 matrix [n, d]).
    """
    try:
        df = pd.read_csv(path, header=None)
        if df.shape[1] > 2 and np.issubdtype(df.dtypes[1], np.number):
            names = df.iloc[:, 0].astype(str)
            X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        else:
            # fallback to headered format
            df = pd.read_csv(path)
            name_col = df.columns[0]
            names = df[name_col].astype(str)
            X = df.drop(columns=[name_col]).to_numpy(dtype=np.float32)
    except Exception:
        # robust final fallback (headered)
        df = pd.read_csv(path)
        name_col = df.columns[0]
        names = df[name_col].astype(str)
        X = df.drop(columns=[name_col]).to_numpy(dtype=np.float32)

    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return pd.Index(names), X


NAME_CLEAN_RE = re.compile(r"[^a-z0-9_]+")

def clean_text(s: str) -> str:
    """
    Language-agnostic normalization:
      - lowercase
      - replace spaces/dashes with underscore
      - strip other punctuation
      - collapse multiple underscores
    """
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = NAME_CLEAN_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# Model building 

def fit_text_to_embedding_mapper(
    names: Iterable[str],
    X_norm: np.ndarray,
    ngram_range=(3, 5),
    min_df=2,
    alpha=1.0,
) -> Tuple[TfidfVectorizer, Ridge]:
    """
    Learn linear map W: TFIDF(names) @ W ≈ X_norm.
    Returns (vectorizer, ridge_model).
    """
    names_clean = [clean_text(n) for n in names]
    vect = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        norm="l2",
    )
    A = vect.fit_transform(names_clean)

    reg = Ridge(alpha=alpha, fit_intercept=False, solver="auto")
    reg.fit(A, X_norm)
    return vect, reg


def encode_query_to_vec(query: str, vectorizer: TfidfVectorizer, reg_model: Ridge) -> np.ndarray:
    """
    Map arbitrary text to the subreddit embedding space via TFIDF -> Ridge.
    """
    q = clean_text(query)
    Aq = vectorizer.transform([q])
    qv = Aq @ reg_model.coef_.T
    qv = np.asarray(qv).ravel().astype(np.float32)
    qv /= (np.linalg.norm(qv) + 1e-12)
    return qv


class SubredditKNN:
    def __init__(self, names: pd.Index, X_norm: np.ndarray, metric: str = "cosine"):
        self.names = names
        self.X = X_norm
        self.nn = NearestNeighbors(n_neighbors=min(50, len(names)), metric=metric)
        self.nn.fit(self.X)

    def search(self, q: np.ndarray, k: int = 25) -> List[tuple[str, float]]:
        n_req = min(max(k + 50, k * 2), len(self.names))
        distances, idxs = self.nn.kneighbors(q.reshape(1, -1), n_neighbors=n_req)
        sims = 1.0 - distances[0]
        out = [(self.names[i], float(sims[j])) for j, i in enumerate(idxs[0])]
        return out[:k]


# Public callable entry point 

def run_knn(
    query: str,
    emb: str,
    k=10,
    alpha=1.0, 
    min_df=2,
    ngram_min=3,
    ngram_max=5,
    debug=False,) -> List[tuple[str, float]]:
    """
    Pure function used from other modules or CLI:
    - query: keyword(s) or phrase
    - emb: path to subreddit embeddings CSV
    - k: number of neighbors to return
    - alpha: ridge regularization
    - min_df: TFIDF min_df for char n-grams
    - ngram_min: char n-gram min size
    - ngram_max: char n-gram max size
    - debug: print basic diagnostics
    - Returns List[(name, similarity)]
    """
    names, X = load_subreddit_embeddings(emb)

    if debug:
        print(f"# Loaded {len(names)} subreddits; dim={X.shape[1]}")
        print(f"# First 5: {list(names[:5])}")

    vect, reg = fit_text_to_embedding_mapper(
        names.tolist(),
        X,
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        alpha=alpha,
    )

    qv = encode_query_to_vec(query, vect, reg)
    knn = SubredditKNN(names, X)
    results = knn.search(qv, k=k)
    return results


# Cached version of run_knn

CACHE_DIR = "data/knn_cache"


def _get_cache_key(query: str, emb: str, k: int, alpha: float, min_df: int, 
                   ngram_min: int, ngram_max: int) -> str:
    """
    Generate a unique cache key based on query parameters.
    """
    params = f"{query}|{emb}|{k}|{alpha}|{min_df}|{ngram_min}|{ngram_max}"
    return hashlib.md5(params.encode()).hexdigest()


def run_knn_cached(
    query: str,
    emb: str,
    k=10,
    alpha=1.0, 
    min_df=2,
    ngram_min=3,
    ngram_max=5,
    debug=False,
    use_cache=True,
    cache_dir=None,
) -> List[tuple[str, float]]:
    """
    Cached version of run_knn - stores results in a JSON file for faster subsequent runs.
    
    Additional args:
    - use_cache: If True (default), try to load from cache first. Set to False to force recompute.
    - cache_dir: Custom cache directory. Defaults to 'data/knn_cache'.
    
    Returns List[(name, similarity)]
    """
    cache_directory = cache_dir or CACHE_DIR
    cache_key = _get_cache_key(query, emb, k, alpha, min_df, ngram_min, ngram_max)
    cache_file = os.path.join(cache_directory, f"{cache_key}.json")
    
    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                if debug:
                    print(f"# Loaded results from cache: {cache_file}")
                # Convert back to list of tuples
                return [(item[0], item[1]) for item in cached_data['results']]
        except (json.JSONDecodeError, KeyError, IOError) as e:
            if debug:
                print(f"# Cache read failed: {e}, recomputing...")
    
    # Compute results using the original function
    results = run_knn(
        query=query,
        emb=emb,
        k=k,
        alpha=alpha,
        min_df=min_df,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        debug=debug,
    )
    
    # Save to cache
    try:
        os.makedirs(cache_directory, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump({
                'query': query,
                'emb': emb,
                'k': k,
                'alpha': alpha,
                'min_df': min_df,
                'ngram_min': ngram_min,
                'ngram_max': ngram_max,
                'results': results,
            }, f, indent=2)
        if debug:
            print(f"# Cached results to: {cache_file}")
    except IOError as e:
        if debug:
            print(f"# Warning: Could not cache results: {e}")
    
    return results


# CLI wrapper 

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="KNN over subreddit embeddings with learned text→embedding projection (no hardcoded rules)."
    )
    ap.add_argument("query", type=str, help="Keyword(s) or phrase (e.g., 'turkey coup')")
    ap.add_argument("--emb", type=str, default="data/web-redditEmbeddings-subreddits.csv",
                    help="Path to subreddit embeddings CSV")
    ap.add_argument("--k", type=int, default=25, help="Number of neighbors to return")
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization")
    ap.add_argument("--min_df", type=int, default=2, help="TFIDF min_df for char n-grams")
    ap.add_argument("--ngram_min", type=int, default=3, help="Char n-gram min size")
    ap.add_argument("--ngram_max", type=int, default=5, help="Char n-gram max size")
    ap.add_argument("--debug", action="store_true", help="Print basic diagnostics")
    return ap


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    results = run_knn(
        query=args.query,
        emb=args.emb,
        k=args.k,
        alpha=args.alpha,
        min_df=args.min_df,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        debug=args.debug,
    )

    print(f"# Top {args.k} nearest subreddits (cosine similarity):")
    for n, s in results:
        print(f"{n}\t{s:.6f}")


if __name__ == "__main__":
    main()
