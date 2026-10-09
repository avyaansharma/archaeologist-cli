import os
import re
import pickle
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

_BM25_CACHE = {}

def tokenize_text(text: Optional[str]) -> List[str]:
    """Tokenizes text into code-aware and Unicode-aware tokens for robust BM25 matching."""
    if not text or not isinstance(text, str):
        return []
    words = re.findall(r'\w+', text)
    tokens = []
    for w in words:
        w_lower = w.lower()
        tokens.append(w_lower)
        # Split snake_case identifiers
        if "_" in w:
            parts = [p.lower() for p in w.split("_") if p]
            tokens.extend(parts)
        # Split camelCase and PascalCase identifiers
        camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+', w)
        if len(camel_parts) > 1:
            tokens.extend([p.lower() for p in camel_parts if p])
    return tokens

class _SafeBM25Unpickler(pickle.Unpickler):
    SAFE_MODULES = {
        "builtins": {"dict", "list", "set", "tuple", "str", "int", "float", "bool", "bytes", "bytearray"},
        "rank_bm25": {"BM25Okapi"},
        "math": {"log"},
        "datetime": {"datetime", "date", "time", "timezone", "timedelta"},
    }

    def find_class(self, module, name):
        if module in self.SAFE_MODULES and name in self.SAFE_MODULES[module]:
            return super().find_class(module, name)
        if module == "numpy" and ("float" in name or "int" in name or "dtype" in name or "ndarray" in name):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden for security")


class BM25Index:
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[dict] = []

    def fit(self, chunks: List[dict]):
        """Fits the BM25 model on a list of chunk dictionaries."""
        self.chunks = chunks
        tokenized_corpus = [tokenize_text(chunk.get("text")) for chunk in chunks if isinstance(chunk, dict)]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    build = fit

    def search(
        self,
        query: Optional[str],
        limit: int = 10,
        file_path: Optional[str] = None,
        source_types: Optional[List[str]] = None,
        is_reverted: Optional[bool] = None,
        repo_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[dict]:
        """Performs a BM25 keyword search with flexible metadata and temporal filters and returns ranked chunks."""
        if not query or not self.bm25 or not self.chunks:
            return []

        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            return []

        from datetime import datetime, timezone
        def _parse_ts(val: Any) -> Optional[datetime]:
            if not val:
                return None
            if isinstance(val, datetime):
                if val.tzinfo is not None:
                    return val.astimezone(timezone.utc).replace(tzinfo=None)
                return val
            if isinstance(val, str):
                s = val.strip()
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                try:
                    dt = datetime.fromisoformat(s)
                    if dt.tzinfo is not None:
                        return dt.astimezone(timezone.utc).replace(tzinfo=None)
                    return dt
                except Exception:
                    try:
                        return datetime.strptime(s[:10], "%Y-%m-%d")
                    except Exception:
                        pass
            return None

        dt_from = _parse_ts(date_from)
        dt_to = _parse_ts(date_to)

        query_set = set(tokenized_query)
        scores = self.bm25.get_scores(tokenized_query)
        
        results = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            effective_score = float(score)
            if effective_score <= 0:
                # D21: Fast pre-filter to skip expensive tokenization if no query term is present
                chunk_raw = chunk.get("text", "").lower()
                if not any(t in chunk_raw for t in query_set):
                    continue
                # BM25Okapi assigns <=0 IDF to terms appearing in >=50% of the corpus.
                # If the chunk text actually contains the query tokens, assign a positive minimum score.
                chunk_tokens = set(tokenize_text(chunk.get("text")))
                overlap = query_set.intersection(chunk_tokens)
                if overlap:
                    effective_score = 0.5 * len(overlap)
                else:
                    continue

            # Temporal filters
            if dt_from or dt_to:
                chunk_dt = _parse_ts(chunk.get("timestamp"))
                if chunk_dt:
                    if dt_from and chunk_dt < dt_from:
                        continue
                    if dt_to and chunk_dt > dt_to:
                        continue

            # Structural repo_id filter
            if repo_id:
                c_repo = chunk.get("repo_id")
                if not c_repo:
                    continue
                c_clean = c_repo.strip().lower()
                target_clean = repo_id.strip().lower()
                if c_clean != target_clean:
                    c_base = c_clean.split("/")[-1]
                    t_base = target_clean.split("/")[-1]
                    if c_base != t_base:
                        continue
            
            # Structural file_path filter matching against chunk.file_paths
            if file_path:
                fp_lower = file_path.lower()
                c_paths = [p.lower() for p in chunk.get("file_paths", [])]
                if not any(fp_lower in p or p.endswith(fp_lower) or fp_lower.endswith(p) for p in c_paths):
                    continue

            if is_reverted is not None and chunk.get("is_reverted") != is_reverted:
                continue
            if source_types and chunk.get("source_type") not in source_types:
                continue

            results.append({
                "score": float(effective_score),
                "chunk": chunk
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def save(self, file_path: str):
        """Saves the fitted BM25 model and chunks to disk with atomic write and cache synchronization."""
        import threading
        data = {
            "bm25": self.bm25,
            "chunks": self.chunks
        }
        dir_name = os.path.dirname(os.path.abspath(file_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        temp_file = f"{file_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(temp_file, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, file_path)
        except Exception:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            raise

        try:
            stat = os.stat(file_path)
            _BM25_CACHE[file_path] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "data": data
            }
        except Exception:
            pass

    def load(self, file_path: str):
        """Loads a fitted BM25 model and chunks from disk with mtime-validated in-memory caching and safe deserialization."""
        if os.path.exists(file_path):
            try:
                stat = os.stat(file_path)
                cached = _BM25_CACHE.get(file_path)
                if cached and cached.get("mtime") == stat.st_mtime and cached.get("size") == stat.st_size:
                    data = cached["data"]
                    self.bm25 = data["bm25"]
                    self.chunks = data["chunks"]
                    return
            except Exception:
                stat = None

            with open(file_path, "rb") as f:
                data = _SafeBM25Unpickler(f).load()
                self.bm25 = data["bm25"]
                self.chunks = data["chunks"]
                if stat:
                    _BM25_CACHE[file_path] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "data": data
                    }
