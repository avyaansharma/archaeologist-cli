import json
import os
import re
import sqlite3
import subprocess
import tempfile
from pathlib import Path


def find_repo_root(start_path: str | None = None) -> Path:
    """Finds the git repository root or returns current working directory."""
    iso = os.environ.get("ARCHAEOLOGIST_ISOLATION_ROOT")
    temp_dir = Path(tempfile.gettempdir()).resolve()

    def _is_within_boundary(p: Path, boundary: Path | None) -> bool:
        if not boundary:
            return True
        return p == boundary or boundary in p.parents

    if start_path:
        current = Path(start_path).resolve()
        if current.is_file():
            current = current.parent
        boundary = Path(iso).resolve() if iso else (temp_dir if temp_dir in current.parents or current == temp_dir else None)

        for parent in [current] + list(current.parents):
            if boundary and not _is_within_boundary(parent, boundary):
                break
            if (parent / ".git").exists():
                return parent

        try:
            proc = subprocess.run(
                ["git", "-C", str(current), "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                top = Path(proc.stdout.strip()).resolve()
                if _is_within_boundary(top, boundary) and (top == current or top in current.parents):
                    return top
        except Exception:
            pass
        return current

    if iso:
        return Path(iso).resolve()

    current = Path(os.getcwd()).resolve()
    boundary = temp_dir if temp_dir in current.parents or current == temp_dir else None

    for parent in [current] + list(current.parents):
        if boundary and not _is_within_boundary(parent, boundary):
            break
        if (parent / ".git").exists():
            return parent

    try:
        proc = subprocess.run(
            ["git", "-C", str(current), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            top = Path(proc.stdout.strip()).resolve()
            if _is_within_boundary(top, boundary) and (top == current or top in current.parents):
                return top
    except Exception:
        pass
    return current


def get_archaeologist_dir(repo_path: str | None = None, create: bool = True) -> Path:
    """Returns the dedicated .archaeologist directory inside the repository."""
    root = find_repo_root(repo_path)
    if root.is_file():
        root = root.parent
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


get_repo_data_dir = get_archaeologist_dir

from archaeologist.storage.context import (
    current_bm25_path_var,
    current_db_url_var,
    current_qdrant_path_var,
    current_repo_path_var,
)


def resolve_repo_id(repo_path: str, repo_url: str | None = None) -> str:
    """Computes canonical repo_id (owner/name or dir name)."""
    base_dir = find_repo_root(repo_path)
    url = repo_url or detect_github_remote(str(base_dir))
    if url:
        parts = url.rstrip("/").removesuffix(".git").split("/")
        if len(parts) >= 2:
            owner, name = parts[-2:]
            return f"{owner}/{name}".lower()
        elif len(parts) == 1:
            return parts[0].lower()
    return base_dir.name.lower()


def save_stored_repo_meta(
    repo_id: str,
    repo_path: str | None = None,
    repo_url: str | None = None,
    embedder_provider: str | None = None,
    embedder_dimension: int | None = None,
    last_ingested_at: str | None = None,
    embedder_model: str | None = None
) -> None:
    """Stores repository metadata in .archaeologist/meta.json and database repo_meta table."""
    try:
        arch_dir = get_archaeologist_dir(repo_path)
        meta_file = arch_dir / "meta.json"
        meta_data = {
            "repo_id": repo_id,
            "repo_url": repo_url,
            "embedder_provider": embedder_provider,
            "embedder_model": embedder_model,
            "embedder_dimension": embedder_dimension,
            "last_ingested_at": last_ingested_at
        }
        meta_file.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
    except Exception:
        pass

    target_db_urls = []
    ctx_url = current_db_url_var.get()
    if ctx_url and ctx_url.startswith("sqlite:///"):
        target_db_urls.append(ctx_url)
    try:
        default_url = get_default_db_url(repo_path)
        if default_url and default_url.startswith("sqlite:///") and default_url not in target_db_urls:
            target_db_urls.append(default_url)
    except Exception:
        pass

    for db_url in target_db_urls:
        try:
            db_path = db_url.split("?")[0].replace("sqlite:///", "")
            conn = sqlite3.connect(db_path)
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS repo_meta (repo_id TEXT PRIMARY KEY, repo_name TEXT, repo_url TEXT, embedder_provider TEXT, embedder_model TEXT, embedder_dimension INTEGER, last_ingested_at TIMESTAMP, key TEXT, value TEXT, created_at TIMESTAMP, updated_at TIMESTAMP)"
                )
                # Ensure columns exist if table was already created
                for col, ctype in [
                    ("embedder_provider", "TEXT"),
                    ("embedder_model", "TEXT"),
                    ("embedder_dimension", "INTEGER"),
                    ("last_ingested_at", "TIMESTAMP"),
                    ("key", "TEXT"),
                    ("value", "TEXT"),
                ]:
                    try:
                        conn.execute(f"ALTER TABLE repo_meta ADD COLUMN {col} {ctype}")
                    except Exception:
                        pass
                conn.execute(
                    """INSERT INTO repo_meta (repo_id, repo_url, embedder_provider, embedder_model, embedder_dimension, last_ingested_at, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                    ON CONFLICT(repo_id) DO UPDATE SET
                        repo_url = COALESCE(excluded.repo_url, repo_meta.repo_url),
                        embedder_provider = COALESCE(excluded.embedder_provider, repo_meta.embedder_provider),
                        embedder_model = COALESCE(excluded.embedder_model, repo_meta.embedder_model),
                        embedder_dimension = COALESCE(excluded.embedder_dimension, repo_meta.embedder_dimension),
                        last_ingested_at = COALESCE(excluded.last_ingested_at, repo_meta.last_ingested_at),
                        updated_at = datetime('now')""",
                    (repo_id, repo_url, embedder_provider, embedder_model, embedder_dimension, last_ingested_at)
                )
            conn.close()
        except Exception:
            pass


def get_stored_repo_meta(repo_path: str | None = None, repo_id: str | None = None) -> dict:
    """Retrieves stored repository metadata from meta.json or SQLite repo_meta table."""
    if repo_path or not current_db_url_var.get():
        try:
            arch_dir = get_archaeologist_dir(repo_path)
            meta_file = arch_dir / "meta.json"
            if meta_file.exists():
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data:
                    if not repo_id or data.get("repo_id") == repo_id:
                        return data
        except Exception:
            pass

    db_url = current_db_url_var.get() or get_default_db_url(repo_path)
    if db_url and db_url.startswith("sqlite:///"):
        cand_db = db_url.split("?")[0].replace("sqlite:///", "")
        if os.path.exists(cand_db):
            try:
                conn = sqlite3.connect(cand_db)
                cursor = conn.cursor()
                if repo_id:
                    cursor.execute(
                        "SELECT repo_id, repo_url, embedder_provider, embedder_model, embedder_dimension, last_ingested_at FROM repo_meta WHERE repo_id = ? LIMIT 1",
                        (repo_id,),
                    )
                else:
                    cursor.execute(
                        "SELECT repo_id, repo_url, embedder_provider, embedder_model, embedder_dimension, last_ingested_at FROM repo_meta LIMIT 1"
                    )
                row = cursor.fetchone()
                conn.close()
                if row:
                    return {
                        "repo_id": row[0],
                        "repo_url": row[1],
                        "embedder_provider": row[2],
                        "embedder_model": row[3],
                        "embedder_dimension": row[4],
                        "last_ingested_at": row[5],
                    }
            except Exception:
                pass
    return {}


def get_stored_repo_id(repo_path: str | None = None, db_path: str | None = None) -> str | None:
    """Retrieves stored repo_id from meta.json, SQLite repo_meta table, or database tables."""
    direct_db = db_path
    if not direct_db and repo_path and (repo_path.endswith(".db") or os.path.isfile(repo_path)):
        direct_db = repo_path
        repo_path = None

    if direct_db and os.path.exists(direct_db):
        try:
            conn = sqlite3.connect(direct_db)
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT repo_id FROM repo_meta WHERE repo_id IS NOT NULL AND repo_id != '' LIMIT 1")
                row = cursor.fetchone()
                if row and row[0]:
                    conn.close()
                    return row[0]
            except Exception:
                pass
            try:
                cursor.execute("SELECT repo_id FROM [commit] WHERE repo_id IS NOT NULL AND repo_id != '' LIMIT 1")
                row = cursor.fetchone()
                if row and row[0]:
                    conn.close()
                    return row[0]
            except Exception:
                pass
            conn.close()
        except Exception:
            pass

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
            cand_db = db_url.split("?")[0].replace("sqlite:///", "")
            if os.path.exists(cand_db):
                conn = sqlite3.connect(cand_db)
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT repo_id FROM repo_meta WHERE repo_id IS NOT NULL AND repo_id != '' LIMIT 1")
                    row = cursor.fetchone()
                    if row and row[0]:
                        conn.close()
                        return row[0]
                except Exception:
                    pass
                try:
                    cursor.execute("SELECT repo_id FROM [commit] WHERE repo_id IS NOT NULL AND repo_id != '' LIMIT 1")
                    row = cursor.fetchone()
                    if row and row[0]:
                        conn.close()
                        return row[0]
                except Exception:
                    pass
                conn.close()
    except Exception:
        pass
    return None


def get_default_db_path(repo_path: str | None = None) -> str:
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


def get_default_db_url(repo_path: str | None = None) -> str:
    """Returns the default SQLite SQLAlchemy connection URL. repo_path > ContextVar > ARCHAEOLOGIST_DB_URL > SQLite DATABASE_URL."""
    if repo_path:
        path_str = str(get_archaeologist_dir(repo_path) / "archaeologist.db").replace("\\", "/")
        return f"sqlite:///{path_str}"

    ctx_url = current_db_url_var.get()
    if ctx_url:
        return ctx_url

    arch_url = os.getenv("ARCHAEOLOGIST_DB_URL")
    if arch_url:
        return arch_url

    custom_url = os.getenv("DATABASE_URL")
    if custom_url and custom_url.startswith("sqlite"):
        return custom_url

    path_str = get_default_db_path(repo_path).replace("\\", "/")
    return f"sqlite:///{path_str}"


def get_default_bm25_path(repo_path: str | None = None) -> str:
    """Returns default BM25 index path. repo_path > ContextVar > ARCHAEOLOGIST_BM25_PATH > BM25_INDEX_PATH."""
    if repo_path:
        return str(get_archaeologist_dir(repo_path) / "bm25_index.bin")

    ctx_bm25 = current_bm25_path_var.get()
    if ctx_bm25:
        return ctx_bm25

    arch_bm25 = os.getenv("ARCHAEOLOGIST_BM25_PATH")
    if arch_bm25:
        return arch_bm25

    custom = os.getenv("BM25_INDEX_PATH")
    if custom:
        return custom

    return str(get_archaeologist_dir(repo_path) / "bm25_index.bin")


def get_default_qdrant_path(repo_path: str | None = None) -> str:
    """Returns default embedded Qdrant storage path. repo_path > ContextVar > QDRANT_STORAGE_PATH > ARCHAEOLOGIST_REPO > cwd."""
    if repo_path:
        return str(get_archaeologist_dir(repo_path) / "qdrant_db")

    ctx_qdrant = current_qdrant_path_var.get()
    if ctx_qdrant:
        return ctx_qdrant

    custom = os.getenv("QDRANT_STORAGE_PATH")
    if custom:
        return custom

    ctx_repo = current_repo_path_var.get() or os.getenv("ARCHAEOLOGIST_REPO")
    if ctx_repo:
        return str(get_archaeologist_dir(ctx_repo) / "qdrant_db")

    return str(get_archaeologist_dir(repo_path) / "qdrant_db")


def detect_github_remote(repo_path: str | None = None) -> str | None:
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
                return url.rstrip("/").removesuffix(".git").rstrip("/")
            if url.startswith("http://github.com/"):
                return "https://" + url[7:].rstrip("/").removesuffix(".git").rstrip("/")
    except Exception:
        pass
    return None


def calculate_window_since(window: str | None) -> str | None:
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
