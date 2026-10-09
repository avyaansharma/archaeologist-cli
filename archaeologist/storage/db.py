import os
import sys
import threading
from contextlib import contextmanager
from typing import Generator, Optional, Dict

from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

from archaeologist.storage.paths import get_default_db_url
from archaeologist.storage.context import current_db_url_var
from archaeologist.storage.models import Commit, PullRequest, Issue, Chunk, SymbolIndex

load_dotenv()

_DB_LOCK = threading.RLock()
_ENGINES: Dict[str, Engine] = {}

def _create_engine_for_url(url: str) -> Engine:
    if url.startswith("sqlite") and ":memory:" not in url:
        raw_path = url.split("?")[0].replace("sqlite:///", "")
        if raw_path:
            db_dir = os.path.dirname(os.path.abspath(raw_path))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
    eng = create_engine(
        url,
        echo=False,
        connect_args={"check_same_thread": False} if url.startswith("sqlite") else {},
        pool_pre_ping=True
    )
    if url.startswith("sqlite") and ":memory:" not in url:
        from sqlalchemy import event
        @event.listens_for(eng, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            try:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.close()
            except Exception:
                pass

    SQLModel.metadata.create_all(eng)
    if url.startswith("sqlite") and ":memory:" not in url:
        try:
            with eng.connect() as conn:
                for table in ["commit", "pullrequest", "issue", "chunk", "symbolindex", "repo_meta"]:
                    try:
                        conn.exec_driver_sql(f'ALTER TABLE "{table}" ADD COLUMN repo_id VARCHAR;')
                    except Exception:
                        pass
                conn.commit()
        except Exception as e:
            print(f"Notice: Could not migrate columns: {e}", file=sys.stderr)
    return eng

def get_engine_for_url(url: str) -> Engine:
    """Returns or creates a cached Engine for the specified database URL in a thread-safe manner."""
    with _DB_LOCK:
        if url in _ENGINES:
            return _ENGINES[url]
        eng = _create_engine_for_url(url)
        _ENGINES[url] = eng
        return eng


def init_db(db_url: Optional[str] = None):
    """Initializes the database schema and enables SQLite WAL mode for high concurrency."""
    target_url = db_url or get_default_db_url()
    if db_url:
        old_url = get_default_db_url()
        current_db_url_var.set(db_url)
        if old_url and old_url != db_url and old_url in _ENGINES:
            try:
                _ENGINES[old_url].dispose()
                del _ENGINES[old_url]
            except Exception:
                pass
    return get_engine_for_url(target_url)


def dispose_all_engines():
    """Disposes all cached engines cleanly on shutdown."""
    with _DB_LOCK:
        for u, eng in list(_ENGINES.items()):
            try:
                eng.dispose()
            except Exception:
                pass
        _ENGINES.clear()


def get_session(db_url: Optional[str] = None) -> Session:
    """Returns a new SQLModel database session, dynamically syncing with active or specified db_url."""
    target_url = db_url or get_default_db_url()
    return Session(get_engine_for_url(target_url))


@contextmanager
def get_session_context(db_url: Optional[str] = None) -> Generator[Session, None, None]:
    """Context manager yielding a session with connection pooling and automatic rollback."""
    target_url = db_url or get_default_db_url()
    active_engine = get_engine_for_url(target_url)
    session = Session(active_engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def __getattr__(name: str):
    if name == "engine":
        return get_engine_for_url(get_default_db_url())
    if name == "DATABASE_URL":
        return get_default_db_url()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


