import datetime
import math
import re
from typing import List, Dict, Any, Optional

def reciprocal_rank_fusion(
    dense_hits: List[Dict[str, Any]], 
    sparse_hits: List[Dict[str, Any]], 
    k: int = 60,
    limit: int = 10,
    query: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Applies Time-Decay Reciprocal Rank Fusion (RRF) to merge vector and keyword search results.
    
    Incorporate:
    1. Time-Decay RRF (Option 1): Steeper decay (lambda=0.18) for old commit diffs on architectural queries.
    2. AST Symbol Exact Match Rank Boosting (Option 2): 2.0x score boost for active AST symbols matching query.
    3. Zero-Latency Intent Detection (Option 3): Relax decay when query asks for historical motivation (why/revert/PR).
    """
    rrf_scores: Dict[str, float] = {}
    chunk_payloads: Dict[str, Dict[str, Any]] = {}

    current_year = datetime.datetime.now(datetime.timezone.utc).year

    # Option 3: Zero-Latency Intent Detection
    is_historical_intent = False
    if query:
        hist_pattern = r'\b(why|revert|reverted|pr|issue|blame|sha|commit|author|history|deprecated|legacy|origin|motivation|reason|#\d+)\b'
        if re.search(hist_pattern, query, flags=re.IGNORECASE):
            is_historical_intent = True

    # Option 2: Extract query symbols for AST exact match boosting
    query_symbols = set()
    if query:
        raw_syms = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]{3,})\b', query)
        query_symbols = {s.lower() for s in raw_syms if len(s) >= 4}

    # Helper to clean/standardize payload representation and apply RRF + decay + boost
    def add_hit(chunk_id: str, rank: int, payload: Dict[str, Any], initial_weight: float = 1.0):
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0.0
            chunk_payloads[chunk_id] = payload

        # Base RRF score contribution
        score_contrib = (1.0 / (k + rank)) * initial_weight

        # Option 1: Time-Decay RRF
        source_type = payload.get("source_type", "")
        timestamp_val = payload.get("timestamp")

        decay_factor = 1.0
        if source_type in ("commit", "commit_diff", "historical_commit"):
            commit_year = current_year
            if timestamp_val:
                try:
                    if isinstance(timestamp_val, str):
                        commit_year = int(timestamp_val[:4])
                    elif hasattr(timestamp_val, "year"):
                        commit_year = timestamp_val.year
                except Exception:
                    commit_year = current_year

            years_old = max(0, current_year - commit_year)

            if is_historical_intent:
                decay_lambda = 0.01
            else:
                # Balanced time-decay factor for architectural queries (gentle penalty for ancient commits)
                decay_lambda = 0.03

            decay_factor = math.exp(-decay_lambda * years_old)

        elif source_type in ("source_code", "file", "ast_symbol", "code"):
            decay_factor = 1.50

        # Option 2: AST Symbol Exact Match Rank Boosting
        symbol_boost = 1.0
        symbols_modified = payload.get("symbols_modified") or []
        text_content = payload.get("text", "")
        if query_symbols:
            if isinstance(symbols_modified, list) and symbols_modified:
                lowered_syms = [str(s).lower() for s in symbols_modified]
                if any(qs in s for qs in query_symbols for s in lowered_syms):
                    symbol_boost = 2.0
            elif any(qs in text_content.lower() for qs in query_symbols):
                symbol_boost = 1.25

        rrf_scores[chunk_id] += score_contrib * decay_factor * symbol_boost

    # Process dense (vector) hits sorted by raw score descending
    sorted_dense = sorted(
        dense_hits,
        key=lambda h: float(h.get("score") if h.get("score") is not None else 0.0),
        reverse=True
    )
    for rank, hit in enumerate(sorted_dense, start=1):
        raw_payload = hit.get("payload")
        payload = dict(raw_payload) if isinstance(raw_payload, dict) else {}
        chunk_id = payload.get("id") or payload.get("chunk_id") or hit.get("id") or str(rank)
        if "id" not in payload:
            payload["id"] = chunk_id
        add_hit(chunk_id, rank, payload)

    # Process sparse (BM25) hits sorted by raw score descending
    sorted_sparse = sorted(
        sparse_hits,
        key=lambda h: float(h.get("score") if h.get("score") is not None else 0.0),
        reverse=True
    )
    for rank, hit in enumerate(sorted_sparse, start=1):
        chunk_data = hit.get("chunk") or {}
        if not isinstance(chunk_data, dict):
            chunk_data = {}
        chunk_id = chunk_data.get("id") or str(rank)
        
        ts = chunk_data.get("timestamp")
        ts_iso = ts.isoformat() if (ts and hasattr(ts, "isoformat")) else ts
        
        payload = {
            "id": chunk_id,
            "repo_id": chunk_data.get("repo_id"),
            "source_type": chunk_data.get("source_type", "source_code"),
            "source_id": chunk_data.get("source_id", ""),
            "text": chunk_data.get("text", ""),
            "timestamp": ts_iso,
            "file_paths": chunk_data.get("file_paths", []),
            "symbols_modified": chunk_data.get("symbols_modified", []),
            "related_ids": chunk_data.get("related_ids", []),
            "is_reverted": chunk_data.get("is_reverted", False)
        }
        add_hit(chunk_id, rank, payload)

    # Sort candidates desc by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused_results = []
    for cid in sorted_ids[:limit]:
        fused_results.append({
            "id": cid,
            "rrf_score": rrf_scores[cid],
            "payload": chunk_payloads[cid]
        })

    return fused_results

