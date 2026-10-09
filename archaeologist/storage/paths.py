import os
import re
import subprocess
from pathlib import Path
from typing import Optional


import json
import sqlite3

def find_repo_root(start_path: Optional[str] = None) -> Path:
    """Finds the git repository root or returns current working directory."""
    current = Path(start_path or os.getcwd()).resolve()
    try:
        proc = subprocess.run(
            ["git", "-C", str(current), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return Path(proc.stdout.strip()).resolve()
    except Exception:
        pass
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    return current


def get_archaeologist_dir(repo_path: Optional[str] = None, create: bool = True) -> Path:
    """Returns the dedicated .archaeologist directory inside the repository."""
    root = find_repo_root(repo_path)
    arch_dir = root / ".archaeologist"
    if create:
        arch_dir.mkdir(parents=True, exist_ok=True)
        gi = arch_dir / ".gitignore"
        if not gi.exists():
            try:
                gi.write_text("*\n", encoding="utf-8")
            except Exception:
                pass
    return arch_dir


from archaeologist.storage.context import current_db_url_var, current_bm25_path_var


def resolve_repo_id(repo_path: str, repo_url: Optional[str] = None) -> str:
    """Computes canonical repo_id (owner/name or dir name)."""
    url = repo_url or detect_github_remote(repo_path)
    if url:
        parts = url.rstrip("/").removesuffix(".git").split("/")
        if len(parts) >= 2:
            owner, name = parts[-2:]
            return f"{owner}/{name}".lower()
        elif len(parts) == 1:
            return parts[0].lower()
    return Path(repo_path).resolve().name.lower()


def save_stored_repo_meta(repo_id: str, repo_path: Optional[str] = None, repo_url: Optional[str] = None) -> None:
    """Stores repository metadata in .archaeologist/meta.json and database repo_meta table."""
    try:
        arch_dir = get_archaeologist_dir(repo_path)
        meta_file = arch_dir / "meta.json"
        meta_data = {"repo_id": repo_id, "repo_url": repo_url}
        meta_file.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
    except Exception:
        pass

    try:
        db_url = get_default_db_url(repo_path)
        if db_url.startswith("sqlite:///"):
            db_path = db_url.split("?")[0].replace("sqlite:///", "")
            conn = sqlite3.connect(db_path)
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS repo_meta (repo_id TEXT PRIMARY KEY, repo_name TEXT, repo_url TEXT, created_at TIMESTAMP, updated_at TIMESTAMP)"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO repo_meta (repo_id, repo_url) VALUES (?, ?)",
                    (repo_id, repo_url)
                )
            conn.close()
    except Exception:
        pass


def get_stored_repo_id(repo_path: Optional[str] = None) -> Optional[str]:
    """Retrieves stored repo_id from meta.json or SQLite repo_meta table."""
    try:
        arch_dir = get_archaeologist_dir(repo_path)
        meta_file = arch_dir / "meta.json"
        if meta_file.exists():
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            if data.get("repo_id"):
                return data["repo_id"]
    except Exception:
        pass

    try:
        db_url = get_default_db_url(repo_path)
        if db_url.startswith("sqlite:///"):
            db_path = db_url.split("?")[0].replace("sqlite:///", "")
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT repo_id FROM repo_meta LIMIT 1")
                row = cursor.fetchone()
                conn.close()
                if row and row[0]:
                    return row[0]
    except Exception:
        pass
    return None


def get_default_db_path(repo_path: Optional[str] = None) -> str:
    """Returns the default SQLite database path."""
    ctx_url = current_db_url_var.get()
    if ctx_url and ctx_url.startswith("sqlite:///"):
        return ctx_url.split("?")[0].replace("sqlite:///", "")

    if not repo_path:
        arch_url = os.getenv("ARCHAEOLOGIST_DB_URL")
        if arch_url and arch_url.startswith("sqlite:///"):
            return arch_url.split("?")[0].replace("sqlite:///", "")
        custom_url = os.getenv("DATABASE_URL")
        if custom_url and custom_url.startswith("sqlite:///"):
            return custom_url.split("?")[0].replace("sqlite:///", "")

    return str(get_archaeologist_dir(repo_path) / "archaeologist.db")


def get_default_db_url(repo_path: Optional[str] = None) -> str:
    """Returns the default SQLite SQLAlchemy connection URL. ContextVar > repo_path > ARCHAEOLOGIST_DB_URL > SQLite DATABASE_URL."""
    ctx_url = current_db_url_var.get()
    if ctx_url:
        return ctx_url

    if repo_path:
        path_str = str(get_archaeologist_dir(repo_path) / "archaeologist.db").replace("\\", "/")
        return f"sqlite:///{path_str}"

    arch_url = os.getenv("ARCHAEOLOGIST_DB_URL")
    if arch_url:
        return arch_url

    custom_url = os.getenv("DATABASE_URL")
    if custom_url and custom_url.startswith("sqlite"):
        return custom_url

    path_str = get_default_db_path(repo_path).replace("\\", "/")
    return f"sqlite:///{path_str}"


def get_default_bm25_path(repo_path: Optional[str] = None) -> str:
    """Returns default BM25 index path. ContextVar > repo_path > ARCHAEOLOGIST_BM25_PATH > BM25_INDEX_PATH."""
    ctx_bm25 = current_bm25_path_var.get()
    if ctx_bm25:
        return ctx_bm25

    if repo_path:
        return str(get_archaeologist_dir(repo_path) / "bm25_index.bin")

    arch_bm25 = os.getenv("ARCHAEOLOGIST_BM25_PATH")
    if arch_bm25:
        return arch_bm25

    custom = os.getenv("BM25_INDEX_PATH")
    if custom:
        return custom

    return str(get_archaeologist_dir(repo_path) / "bm25_index.bin")


def get_default_qdrant_path(repo_path: Optional[str] = None) -> str:
    """Returns default embedded Qdrant storage path."""
    if not repo_path:
        custom = os.getenv("QDRANT_STORAGE_PATH")
        if custom:
            return custom
    return str(get_archaeologist_dir(repo_path) / "qdrant_db")


def detect_github_remote(repo_path: Optional[str] = None) -> Optional[str]:
    """Extracts GitHub https URL from git remote origin if available."""
    root = find_repo_root(repo_path)
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode == 0:
            url = proc.stdout.strip()
            # Convert git@github.com:owner/repo.git or ssh://git@github.com/owner/repo.git
            ssh_match = re.match(r"^(?:ssh://)?git@github\.com[:/]([^/]+)/(.+?)(?:\.git)?/?$", url)
            if ssh_match:
                return f"https://github.com/{ssh_match.group(1)}/{ssh_match.group(2)}"
            if url.startswith("https://github.com/"):
                return url.removesuffix(".git").removesuffix("/")
    except Exception:
        pass
    return None


def calculate_window_since(window: Optional[str]) -> Optional[str]:
    """Calculates the ISO YYYY-MM-DD cutoff date from a window descriptor.

    Supported windows:
    - '6m', '6months', '6-months': ~6 months (182 days)
    - '1y', '1year', '1-year': 1 year (365 days)
    - '2y', '2years', '2-years': 2 years (730 days)
    - 'full', 'all': Full repository history (returns None)
    """
    if not window:
        return None
    normalized = window.strip().lower().replace(" ", "").replace("-", "")
    from datetime import datetime, timedelta
    now = datetime.now()
    if normalized in ("6m", "6months", "6month"):
        return (now - timedelta(days=182)).strftime("%Y-%m-%d")
    elif normalized in ("1y", "1year", "1years"):
        return (now - timedelta(days=365)).strftime("%Y-%m-%d")
    elif normalized in ("2y", "2year", "2years"):
        return (now - timedelta(days=730)).strftime("%Y-%m-%d")
    elif normalized in ("full", "all", "none"):
        return None
    else:
        raise ValueError(
            f"Invalid window '{window}'. Allowed options: '6m' (6 months), '1y' (1 year), '2y' (2 years), or 'full' (all history)."
        )
