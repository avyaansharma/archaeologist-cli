import datetime
import math
import re
from typing import Any


def reciprocal_rank_fusion(
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    exact_hits: list[dict[str, Any]] | None = None,
    k: int = 60,
    limit: int = 10,
    query: str | None = None
) -> list[dict[str, Any]]:
    """Applies Time-Decay Reciprocal Rank Fusion (RRF) to merge vector and keyword search results.
    
    Incorporate:
    1. Time-Decay RRF (Option 1): Mild decay (lambda=0.03 default, lambda=0.01 for historical intent, floor=0.35) for older commit diffs.
    2. AST Symbol Exact Match Rank Boosting (Option 2): 1.75x score boost for active AST symbols matching query.
    3. Zero-Latency Intent Detection (Option 3): Relax decay when query asks for historical motivation (why/revert/PR).
    """
    rrf_scores: dict[str, float] = {}
    chunk_payloads: dict[str, dict[str, Any]] = {}

    current_year = datetime.datetime.now(datetime.UTC).year

    # Option 3: Zero-Latency Intent Detection
    is_historical_intent = False
    if query:
        hist_pattern = r'\b(why|revert|reverted|pr|issue|blame|sha|commit|author|history|deprecated|legacy|origin|motivation|reason|#\d+)\b'
        if re.search(hist_pattern, query, flags=re.IGNORECASE):
            is_historical_intent = True

    # Option 2: Extract query symbols for AST exact match boosting
    COMMON_STOPWORDS = {
        "why", "how", "what", "when", "where", "which", "with", "that", "this",
        "from", "into", "were", "been", "have", "some", "more", "make", "made",
        "here", "there", "then", "them", "they", "will", "would", "could", "should",
        "does", "done", "about", "after", "before", "added", "removed", "fixed",
        "changed", "change", "updating", "updated", "modify", "modified", "the",
        "and", "for", "also", "over", "under", "again", "only", "such", "handler"
    }
    query_symbols = set()
    ident_symbols = set()
    if query:
        raw_syms = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b', query)
        for s in raw_syms:
            s_low = s.lower()
            if len(s) >= 3 and s_low not in COMMON_STOPWORDS:
                query_symbols.add(s_low)
                if "_" in s or any(c.isupper() for c in s):
                    ident_symbols.add(s_low)

    # Helper to clean/standardize payload representation and apply RRF + decay + boost
    def add_hit(chunk_id: str, rank: int, payload: dict[str, Any], initial_weight: float = 1.0):
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
                decay_lambda = 0.03

            # Ensure decay factor has a minimum floor (0.35) so older commits are never zeroed out
            decay_factor = max(0.35, math.exp(-decay_lambda * years_old))

        elif source_type in ("source_code", "file", "ast_symbol", "code"):
            # Current code never decays with age
            decay_factor = 1.25

        elif source_type in ("pull_request", "pr", "issue"):
            # PRs and issues carry high context and parity with commits
            decay_factor = 1.0

        # Option 2: AST Symbol Exact Match Rank Boosting
        symbol_boost = 1.0
        symbols_modified = payload.get("symbols_modified") or []
        text_content = payload.get("text", "")
        if query_symbols:
            bare_symbols = set()
            if isinstance(symbols_modified, list):
                for s in symbols_modified:
                    clean = str(s).strip().lower()
                    bare_symbols.add(clean)
                    if "::" in clean:
                        for p in clean.split("::"):
                            bare_symbols.add(p)
                    if "." in clean:
                        for p in clean.split("."):
                            bare_symbols.add(p)

            # Exact match on qualified or bare symbol name (prevents loose substring collisions)
            if any(qs in bare_symbols for qs in query_symbols):
                symbol_boost = 1.75
            elif ident_symbols and any(re.search(rf'\b{re.escape(qs)}\b', text_content, re.IGNORECASE) for qs in ident_symbols):
                symbol_boost = 1.15

        rrf_scores[chunk_id] += score_contrib * decay_factor * symbol_boost

    # Process channels: Exact matches, Symbol matches, Dense vectors, Sparse BM25, and Fallback hits
    exact_list = []
    symbol_list = []
    dense_vector_list = []
    fallback_list = []

    for hit in (exact_hits or []):
        sc = float(hit.get("score") if hit.get("score") is not None else 0.0)
        if sc >= 3.0:
            exact_list.append(hit)
        else:
            symbol_list.append(hit)

    for hit in dense_hits:
        sc = float(hit.get("score") if hit.get("score") is not None else 0.0)
        cid = str(hit.get("id") or (hit.get("payload") or {}).get("id") or "")
        payload_data = hit.get("payload") or {}
        if sc >= 3.0:
            exact_list.append(hit)
        elif sc >= 2.0:
            symbol_list.append(hit)
        elif (
            payload_data.get("is_fallback")
            or payload_data.get("is_symbol_lookup")
            or cid.startswith("like_")
            or cid.startswith("sym_")
            or payload_data.get("source_type") == "text_fallback"
        ):
            fallback_list.append(hit)
        else:
            dense_vector_list.append(hit)

    # 1. Exact SHA/PR matches channel (weight 2.5)
    sorted_exact = sorted(
        exact_list,
        key=lambda h: float(h.get("score") if h.get("score") is not None else 0.0),
        reverse=True
    )
    for rank, hit in enumerate(sorted_exact, start=1):
        raw_payload = hit.get("payload")
        payload = dict(raw_payload) if isinstance(raw_payload, dict) else {}
        chunk_id = payload.get("id") or payload.get("chunk_id") or hit.get("id") or f"exact_{rank}"
        if "id" not in payload:
            payload["id"] = chunk_id
        add_hit(chunk_id, rank, payload, initial_weight=2.5)

    # 1b. AST symbol matches channel (weight 0.2; add_hit also applies 1.75x symbol_boost)
    sorted_symbol = sorted(
        symbol_list,
        key=lambda h: float(h.get("score") if h.get("score") is not None else 0.0),
        reverse=True
    )
    for rank, hit in enumerate(sorted_symbol, start=1):
        raw_payload = hit.get("payload")
        payload = dict(raw_payload) if isinstance(raw_payload, dict) else {}
        chunk_id = payload.get("id") or payload.get("chunk_id") or hit.get("id") or f"symbol_{rank}"
        if "id" not in payload:
            payload["id"] = chunk_id
        add_hit(chunk_id, rank, payload, initial_weight=0.2)

    # 2. Dense vector channel (weight 1.0, or 3.2 for top-1 high-confidence cosine match >= 0.98)
    sorted_dense = sorted(
        dense_vector_list,
        key=lambda h: float(h.get("score") if h.get("score") is not None else 0.0),
        reverse=True
    )
    for rank, hit in enumerate(sorted_dense, start=1):
        raw_payload = hit.get("payload")
        payload = dict(raw_payload) if isinstance(raw_payload, dict) else {}
        chunk_id = payload.get("id") or payload.get("chunk_id") or hit.get("id") or f"dense_{rank}"
        if "id" not in payload:
            payload["id"] = chunk_id
        sc = float(hit.get("score") if hit.get("score") is not None else 0.0)
        dense_weight = 3.2 if (rank == 1 and 0.98 <= sc < 1.0) else 1.0
        add_hit(chunk_id, rank, payload, initial_weight=dense_weight)

    # 3. Sparse (BM25) channel (weight 1.0)
    sorted_sparse = sorted(
        sparse_hits,
        key=lambda h: float(h.get("score") if h.get("score") is not None else 0.0),
        reverse=True
    )
    for rank, hit in enumerate(sorted_sparse, start=1):
        chunk_data = hit.get("chunk") or {}
        if not isinstance(chunk_data, dict):
            chunk_data = {}
        chunk_id = chunk_data.get("id") or f"sparse_{rank}"

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
        add_hit(chunk_id, rank, payload, initial_weight=1.0)

    # 4. Fallback keyword channel (weight 0.2 to prevent burying genuine vector hits)
    for rank, hit in enumerate(fallback_list, start=1):
        raw_payload = hit.get("payload")
        payload = dict(raw_payload) if isinstance(raw_payload, dict) else {}
        chunk_id = payload.get("id") or payload.get("chunk_id") or hit.get("id") or f"fallback_{rank}"
        if "id" not in payload:
            payload["id"] = chunk_id
        add_hit(chunk_id, rank, payload, initial_weight=0.2)

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

