import json
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
from sqlmodel import select
from sqlalchemy import func

from archaeologist.storage.db import get_session_context
from archaeologist.storage.models import Commit, SymbolIndex
from archaeologist.utils.security import escape_like


def _get_json_list(value: Any) -> List[str]:
    """Safely deserializes JSON list fields stored in SQLite."""
    if not value:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return []
    return value


def _apply_repo_id_filter(query, model_col, repo_id: Optional[str]):
    if not repo_id:
        return query
    clean = repo_id.strip()
    clean_lower = clean.lower()
    base = clean.split("/")[-1]
    base_lower = clean_lower.split("/")[-1]
    escaped_base = escape_like(base)
    escaped_base_lower = escape_like(base_lower)
    return query.where(
        (model_col == clean) |
        (func.lower(model_col) == clean_lower) |
        (model_col.like(f"%/{escaped_base}", escape="\\")) |
        (func.lower(model_col).like(f"%/{escaped_base_lower}", escape="\\")) |
        (model_col == base) |
        (func.lower(model_col) == base_lower)
    )


def repo_hotspots_tool(top_n: int = 15, repo_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Calculates top churn files in the repo ranked by commit count."""
    file_counts = Counter()
    with get_session_context() as session:
        query = select(Commit.files_changed)
        query = _apply_repo_id_filter(query, Commit.repo_id, repo_id)
        results = session.exec(query).all()
        for files_changed in results:
            files = _get_json_list(files_changed)
            for f in files:
                file_counts[f] += 1
    
    hotspots = []
    for fpath, count in file_counts.most_common(top_n):
        hotspots.append({"file_path": fpath, "commit_count": count})
    return hotspots


def repo_ownership_tool(file_path: Optional[str] = None, repo_id: Optional[str] = None) -> Dict[str, Any]:
    """Calculates author percentage contribution distribution and bus factor risk."""
    author_counts = Counter()
    file_author_counts = defaultdict(Counter)
    total_commits = 0
    with get_session_context() as session:
        query = select(Commit.author_name, Commit.author_email, Commit.files_changed)
        query = _apply_repo_id_filter(query, Commit.repo_id, repo_id)
        results = session.exec(query).all()
        author_display_names = {}
        for author_name, author_email, files_changed in results:
            author_key = author_email if author_email else (author_name or "unknown")
            if author_name and author_key not in author_display_names:
                author_display_names[author_key] = author_name
            files = _get_json_list(files_changed)
            if not file_path or file_path in files:
                author_counts[author_key] += 1
                total_commits += 1
            for f in files:
                file_author_counts[f][author_key] += 1

    distribution = {}
    for author_key, count in author_counts.items():
        disp_name = author_display_names.get(author_key, author_key)
        distribution[disp_name] = {
            "email": author_key if "@" in author_key else None,
            "commit_count": count,
            "percentage": round((count / total_commits) * 100, 2) if total_commits > 0 else 0.0
        }

    # Calculate bus factor risk (e.g. HIGH if single author owns > 60% of commits)
    max_pct = max([data["percentage"] for data in distribution.values()]) if distribution else 0.0
    risk = "HIGH (Single Author Dominance)" if max_pct > 60.0 else "NORMAL"

    res = {
        "target_file": file_path or "GLOBAL REPOSITORY",
        "total_commits": total_commits,
        "author_distribution": distribution,
        "bus_factor_risk": risk
    }

    if not file_path:
        file_breakdown = {}
        sorted_files = sorted(file_author_counts.items(), key=lambda item: sum(item[1].values()), reverse=True)
        for f, counts in sorted_files[:20]:
            if not counts:
                continue
            f_total = sum(counts.values())
            top_author, top_cnt = counts.most_common(1)[0]
            f_pct = round((top_cnt / f_total) * 100, 2) if f_total > 0 else 0
            file_breakdown[f] = {
                "total_commits": f_total,
                "top_author": top_author,
                "top_author_pct": f_pct,
                "bus_factor_risk": "HIGH" if f_pct > 60.0 else "NORMAL"
            }
        res["per_file_breakdown"] = file_breakdown

    return res


def change_coupling_tool(min_co_commits: int = 2, top_n: int = 15, repo_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Identifies pairs of files that frequently change together in the same commit."""
    pair_counts = Counter()
    with get_session_context() as session:
        query = select(Commit.files_changed)
        query = _apply_repo_id_filter(query, Commit.repo_id, repo_id)
        results = session.exec(query).all()
        for files_changed in results:
            files = sorted(list(set(_get_json_list(files_changed))))
            if len(files) > 25 or len(files) < 2:
                continue
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    pair_counts[(files[i], files[j])] += 1

    couplings = []
    for (f1, f2), count in pair_counts.most_common(top_n):
        if count >= min_co_commits:
            couplings.append({
                "file_a": f1,
                "file_b": f2,
                "co_commit_count": count
            })
    return couplings


def repo_symbols_tool(top_n: int = 20, repo_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists extracted AST code symbols (classes, functions, methods) ranked by modification frequency."""
    with get_session_context() as session:
        query = select(SymbolIndex).order_by(SymbolIndex.commit_count.desc())
        query = _apply_repo_id_filter(query, SymbolIndex.repo_id, repo_id)
        symbols = session.exec(query.limit(top_n)).all()
        return [
            {
                "symbol_id": s.symbol_id,
                "file_path": s.file_path,
                "symbol_name": s.symbol_name,
                "kind": s.kind,
                "commit_count": s.commit_count
            }
            for s in symbols
        ]


def symbol_history_tool(symbol_query: str, repo_id: Optional[str] = None, fuzzy: bool = False) -> List[Dict[str, Any]]:
    """Retrieves all commits that modified a specific AST Code Symbol (e.g. 'AuthService' or 'login')."""
    matching_commits = []
    with get_session_context() as session:
        escaped_query = escape_like(symbol_query)
        query = select(
            Commit.sha,
            Commit.author_name,
            Commit.authored_date,
            Commit.message,
            Commit.symbols_modified,
            Commit.files_changed
        ).where(Commit.symbols_modified.like(f"%{escaped_query}%", escape="\\"))
        query = _apply_repo_id_filter(query, Commit.repo_id, repo_id)
            
        commits = session.exec(query).all()
        sq_lower = symbol_query.lower()
        for sha, author, date, message, syms_raw, files_raw in commits:
            syms = _get_json_list(syms_raw)
            matched = False
            for s in syms:
                s_lower = s.lower()
                final_segment = s_lower.split(":")[-1]
                if fuzzy:
                    if sq_lower in s_lower:
                        matched = True
                        break
                else:
                    if final_segment == sq_lower or s_lower == sq_lower or s_lower.endswith(f"::{sq_lower}"):
                        matched = True
                        break
            if matched:
                matching_commits.append({
                    "sha": sha,
                    "author": author,
                    "date": date.isoformat() if hasattr(date, "isoformat") else str(date),
                    "message": message,
                    "symbols": syms,
                    "files": _get_json_list(files_raw)
                })
    return matching_commits
