from contextvars import ContextVar

# Request/coroutine-scoped execution context for async web servers and thread pools.
# Prevents cross-tenant state leakage without mutating process-global os.environ.
current_repo_id_var: ContextVar[str | None] = ContextVar("current_repo_id", default=None)
current_db_url_var: ContextVar[str | None] = ContextVar("current_db_url", default=None)
current_bm25_path_var: ContextVar[str | None] = ContextVar("current_bm25_path", default=None)
current_qdrant_path_var: ContextVar[str | None] = ContextVar("current_qdrant_path", default=None)
current_repo_path_var: ContextVar[str | None] = ContextVar("current_repo_path", default=None)
current_client_api_key_var: ContextVar[str | None] = ContextVar("current_client_api_key", default=None)
