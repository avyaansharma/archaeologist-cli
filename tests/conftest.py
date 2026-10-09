import pytest
from archaeologist.storage.context import (
    current_repo_id_var,
    current_db_url_var,
    current_bm25_path_var,
    current_client_api_key_var,
)
from archaeologist.storage.db import dispose_all_engines

@pytest.fixture(autouse=True)
def clean_context_and_engines():
    """Autouse fixture ensuring ContextVars and database engines do not leak across tests."""
    t1 = current_repo_id_var.set(None)
    t2 = current_db_url_var.set(None)
    t3 = current_bm25_path_var.set(None)
    t4 = current_client_api_key_var.set(None)
    yield
    current_repo_id_var.reset(t1)
    current_db_url_var.reset(t2)
    current_bm25_path_var.reset(t3)
    current_client_api_key_var.reset(t4)
    dispose_all_engines()
