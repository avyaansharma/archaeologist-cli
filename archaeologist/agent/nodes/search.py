import os
import sys
import re
from typing import Optional, List, Dict, Any
from sqlmodel import select
from archaeologist.agent.state import AgentState
from archaeologist.retrieval.embedder import Embedder
from archaeologist.retrieval.vector_store import VectorStore
from archaeologist.retrieval.bm25_index import BM25Index
from archaeologist.retrieval.fusion import reciprocal_rank_fusion
from archaeologist.storage.db import get_session_context
from archaeologist.storage.models import Chunk

_EMBEDDER_INSTANCE = None
_VECTOR_STORE_INSTANCE = None

SYMBOL_STOPWORDS = {
    "http", "httpx", "python", "code", "file", "path", "test", "class", "func", "defs", 
    "does", "how", "what", "where", "when", "why", "with", "from", "into", "request", 
    "response", "session", "context", "global", "system", "method", "function", 
    "module", "script", "option", "parameter", "config", "configuration", "default", 
    "structure", "pattern", "design", "decision", "rationale", "implementation", 
    "architecture", "hierarchy", "unify", "platform", "error", "handler", "interface", 
    "server", "client", "common", "shared", "core", "base", "main", "init", "start", 
    "stop", "close", "read", "write", "open", "save", "load", "update", "delete"
}

def _get_embedder():
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        _EMBEDDER_INSTANCE = Embedder()
    return _EMBEDDER_INSTANCE

def _get_vector_store(dim: int, storage_path: Optional[str] = None):
    global _VECTOR_STORE_INSTANCE
    from archaeologist.storage.paths import get_default_qdrant_path
    from archaeologist.storage.context import current_qdrant_path_var
    target_path = storage_path or current_qdrant_path_var.get() or os.getenv("QDRANT_STORAGE_PATH") or get_default_qdrant_path()
    if _VECTOR_STORE_INSTANCE is None or getattr(_VECTOR_STORE_INSTANCE, "storage_path", None) != target_path:
        _VECTOR_STORE_INSTANCE = VectorStore(vector_size=dim, storage_path=target_path)
        _VECTOR_STORE_INSTANCE.init_collection()
    return _VECTOR_STORE_INSTANCE

def sanitize_search_query(query: str) -> str:
    """Strips GitHub search qualifiers and git CLI flags to prevent BM25 string matching pollution."""
    cleaned = re.sub(r'(?:repo|is|owner|org|site|author|label):\S+', '', query, flags=re.IGNORECASE)
    cleaned = re.sub(r'git\s+log|--(?:grep|oneline|author|since|until|path|name-only|stat)\S*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'["\'\-\-]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else query

def _apply_repo_filter(stmt, model_col, repo_id: Optional[str]):
    if not repo_id:
        return stmt
    clean_repo = repo_id.strip()
    clean_lower = clean_repo.lower()
    base_repo = clean_repo.split("/")[-1]
    base_lower = clean_lower.split("/")[-1]
    from archaeologist.utils.security import escape_like
    from sqlalchemy import func
    escaped_base = escape_like(base_repo)
    escaped_base_lower = escape_like(base_lower)
    return stmt.where(
        (model_col == clean_repo) |
        (func.lower(model_col) == clean_lower) |
        (model_col.like(f"%/{escaped_base}", escape="\\")) |
        (func.lower(model_col).like(f"%/{escaped_base_lower}", escape="\\")) |
        (model_col == base_repo) |
        (func.lower(model_col) == base_lower)
    )

def search_node(state: AgentState) -> dict:
    raw_queries = state.get("search_queries", [])
    queries_to_run = raw_queries if raw_queries else [state["question"]]

    plan = state.get("_current_plan", {})
    file_path = plan.get("file_path")
    source_types = plan.get("source_types")
    date_from = plan.get("date_from")
    date_to = plan.get("date_to")
    repo_id = state.get("repo_id") or plan.get("repo_id")

    bm25 = BM25Index()
    from archaeologist.storage.paths import get_default_bm25_path
    from archaeologist.storage.context import current_bm25_path_var
    custom_bm25 = current_bm25_path_var.get() or os.getenv("BM25_INDEX_PATH")
    if custom_bm25:
        bm25_path = os.path.abspath(custom_bm25)
    elif os.path.exists("bm25_index.bin"):
        bm25_path = os.path.abspath("bm25_index.bin")
    else:
        bm25_path = os.path.abspath(get_default_bm25_path())

    has_bm25 = os.path.exists(bm25_path)
    if has_bm25:
        bm25.load(bm25_path)

    embedder = _get_embedder()
    vector_store = _get_vector_store(embedder.dimension)

    all_sparse_hits = []
    all_dense_hits = []
    seen_sparse_ids = set()
    seen_dense_ids = set()

    for raw_q in queries_to_run:
        query = sanitize_search_query(raw_q)
        print(f"Agent: Executing search for query: '{query}'...", file=sys.stderr)

        # 1. Sparse BM25 Search
        if has_bm25:
            s_hits = bm25.search(
                query=query,
                limit=30,
                file_path=file_path,
                source_types=source_types,
                repo_id=repo_id
            )
            for hit in s_hits:
                cid = hit["chunk"]["id"]
                if cid not in seen_sparse_ids:
                    seen_sparse_ids.add(cid)
                    all_sparse_hits.append(hit)

        # 2. Dense Vector Search
        try:
            query_vectors, success_flags = embedder.embed_texts([query], return_success_flags=True)
            if query_vectors and success_flags and success_flags[0]:
                d_hits = vector_store.search_chunks(
                    query_vector=query_vectors[0],
                    limit=30,
                    file_path=file_path,
                    date_from=date_from,
                    date_to=date_to,
                    source_types=source_types,
                    repo_id=repo_id
                )
                for hit in d_hits:
                    cid = hit["id"]
                    if cid not in seen_dense_ids:
                        seen_dense_ids.add(cid)
                        all_dense_hits.append(hit)
            else:
                print(f"Notice: Query embedding returned fallback mock vector. Skipping dense vector search for query: '{query}'", file=sys.stderr)
        except Exception as e:
            print(f"Error in dense search: {e}", file=sys.stderr)

        # 3. Validated Commit SHA & PR/Issue Database Matcher
        sha_candidates = [m for m in re.findall(r'\b([0-9a-fA-F]{7,40})\b', raw_q) if not m.isdigit()]
        pr_candidates = [int(p) for p in re.findall(r'#(\d+)', raw_q)]
        if sha_candidates or pr_candidates:
            from archaeologist.storage.models import is_valid_commit_sha, is_valid_pr_or_issue
            from archaeologist.utils.security import escape_like
            with get_session_context() as session:
                for sha in sha_candidates:
                    if is_valid_commit_sha(session, sha, repo_id=repo_id):
                        escaped_sha = escape_like(sha)
                        stmt = select(Chunk).where(
                            (Chunk.source_id.like(f"%{escaped_sha}%", escape="\\")) | 
                            (Chunk.text.like(f"%{escaped_sha}%", escape="\\"))
                        )
                        stmt = _apply_repo_filter(stmt, Chunk.repo_id, repo_id)
                        stmt = stmt.limit(30)
                        exact_chunks = session.exec(stmt).all()
                        for c in exact_chunks:
                            if c.id not in seen_dense_ids:
                                seen_dense_ids.add(c.id)
                                all_dense_hits.append({
                                    "id": c.id,
                                    "score": 3.0,
                                    "payload": {
                                        "id": c.id,
                                        "repo_id": c.repo_id,
                                        "source_type": c.source_type,
                                        "source_id": c.source_id,
                                        "text": c.text,
                                        "timestamp": c.timestamp.isoformat() if hasattr(c.timestamp, "isoformat") else c.timestamp,
                                        "file_paths": c.file_paths,
                                        "symbols_modified": c.symbols_modified,
                                        "related_ids": c.related_ids,
                                        "is_reverted": c.is_reverted
                                    }
                                })
                for pr_num in pr_candidates:
                    if is_valid_pr_or_issue(session, pr_num, repo_id=repo_id):
                        stmt = select(Chunk).where(
                            (Chunk.related_ids.like(f"%#{pr_num}%", escape="\\")) |
                            (Chunk.text.like(f"%#{pr_num}%", escape="\\")) |
                            (Chunk.text.like(f"%PR #{pr_num}%", escape="\\")) |
                            (Chunk.text.like(f"%Issue #{pr_num}%", escape="\\"))
                        )
                        stmt = _apply_repo_filter(stmt, Chunk.repo_id, repo_id)
                        stmt = stmt.limit(30)
                        exact_chunks = session.exec(stmt).all()
                        for c in exact_chunks:
                            if c.id not in seen_dense_ids:
                                seen_dense_ids.add(c.id)
                                all_dense_hits.append({
                                    "id": c.id,
                                    "score": 3.0,
                                    "payload": {
                                        "id": c.id,
                                        "repo_id": c.repo_id,
                                        "source_type": c.source_type,
                                        "source_id": c.source_id,
                                        "text": c.text,
                                        "timestamp": c.timestamp.isoformat() if hasattr(c.timestamp, "isoformat") else c.timestamp,
                                        "file_paths": c.file_paths,
                                        "symbols_modified": c.symbols_modified,
                                        "related_ids": c.related_ids,
                                        "is_reverted": c.is_reverted
                                    }
                                })

        # 4. AST Symbol Graph Direct Retrieval with SymbolIndex Pre-Verification
        symbol_candidates = [m for m in re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]{3,})\b', query) if m.lower() not in SYMBOL_STOPWORDS]
        if symbol_candidates:
            from archaeologist.storage.models import SymbolIndex
            from archaeologist.utils.security import escape_like
            with get_session_context() as session:
                valid_symbols = set()
                for sym in symbol_candidates:
                    escaped_s = escape_like(sym)
                    stmt_sym = select(SymbolIndex.symbol_name).where(SymbolIndex.symbol_name.like(f"%{escaped_s}%", escape="\\"))
                    stmt_sym = _apply_repo_filter(stmt_sym, SymbolIndex.repo_id, repo_id)
                    if session.exec(stmt_sym).first():
                        valid_symbols.add(sym)

                for sym in valid_symbols:
                    escaped_sym = escape_like(sym)
                    stmt = select(Chunk).where(Chunk.symbols_modified.like(f"%{escaped_sym}%", escape="\\"))
                    stmt = _apply_repo_filter(stmt, Chunk.repo_id, repo_id)
                    stmt = stmt.limit(30)

                    raw_chunks = session.exec(stmt).all()
                    sym_chunks = [
                        c for c in raw_chunks 
                        if any(sym.lower() in str(s).lower() for s in (c.symbols_modified or []))
                    ]

                    for c in sym_chunks:
                        if c.id not in seen_dense_ids:
                            seen_dense_ids.add(c.id)
                            all_dense_hits.append({
                                "id": c.id,
                                "score": 1.5,
                                "payload": {
                                    "id": c.id,
                                    "repo_id": c.repo_id,
                                    "source_type": c.source_type,
                                    "source_id": c.source_id,
                                    "text": c.text,
                                    "timestamp": c.timestamp.isoformat() if hasattr(c.timestamp, "isoformat") else c.timestamp,
                                    "file_paths": c.file_paths,
                                    "symbols_modified": c.symbols_modified,
                                    "related_ids": c.related_ids,
                                    "is_reverted": c.is_reverted
                                }
                            })

    # 5. SQLite Direct Text Fallback for Low Hit Count
    if len(all_dense_hits) < 5:
        with get_session_context() as session:
            from archaeologist.utils.security import escape_like
            for raw_q in queries_to_run:
                keywords = [w.lower() for w in sanitize_search_query(raw_q).split() if len(w) >= 4 and w.lower() not in SYMBOL_STOPWORDS]
                for kw in keywords[:4]:
                    escaped_kw = escape_like(kw)
                    stmt = select(Chunk).where(Chunk.text.like(f"%{escaped_kw}%", escape="\\"))
                    stmt = _apply_repo_filter(stmt, Chunk.repo_id, repo_id)
                    stmt = stmt.limit(10)
                    matching_chunks = session.exec(stmt).all()
                    for c in matching_chunks:
                        if c.id not in seen_dense_ids:
                            seen_dense_ids.add(c.id)
                            all_dense_hits.append({
                                "id": c.id,
                                "score": 1.0,
                                "payload": {
                                    "id": c.id,
                                    "repo_id": c.repo_id,
                                    "source_type": c.source_type,
                                    "source_id": c.source_id,
                                    "text": c.text,
                                    "timestamp": c.timestamp.isoformat() if hasattr(c.timestamp, "isoformat") else c.timestamp,
                                    "file_paths": c.file_paths,
                                    "symbols_modified": c.symbols_modified,
                                    "related_ids": c.related_ids,
                                    "is_reverted": c.is_reverted
                                }
                            })

    # 6. Hybrid Fusion across all accumulated query hits
    primary_query = queries_to_run[0] if queries_to_run else state.get("question", "")
    fused_results = reciprocal_rank_fusion(all_dense_hits, all_sparse_hits, limit=10, query=primary_query)
    print(f"Found {len(fused_results)} relevant history chunks across {len(queries_to_run)} search queries.", file=sys.stderr)

    retrieved = state.get("retrieved_chunks", [])
    evidence_map = state.get("evidence_by_chunk_id", {})

    new_chunks = []
    for hit in fused_results:
        cid = hit["id"]
        if cid not in evidence_map:
            chunk_data = hit["payload"]
            chunk_data["id"] = cid
            chunk_data["rrf_score"] = hit["rrf_score"]
            new_chunks.append(chunk_data)
            evidence_map[cid] = chunk_data

    return {
        "retrieved_chunks": retrieved + new_chunks,
        "evidence_by_chunk_id": evidence_map
    }

