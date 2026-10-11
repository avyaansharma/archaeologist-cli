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


@pytest.fixture
def golden_repo(tmp_path):
    """Creates a temporary git repository with commits including spaces, revert, rename, delete."""
    import subprocess
    repo_dir = tmp_path / "golden_repo"
    repo_dir.mkdir()
    git = ["git", "-c", "user.name=Alice Tester", "-c", "user.email=alice@example.com"]

    # 1. git init
    subprocess.run(git + ["init", "-q", str(repo_dir)], check=True)

    # 2. Commit 1: Initial files including a path with spaces
    docs_dir = repo_dir / "docs"
    docs_dir.mkdir()
    (docs_dir / "file with spaces.txt").write_text("Documentation file with spaces in name\n")
    app_file = repo_dir / "app.py"
    app_file.write_text(
        "class WebServer:\n"
        "    '''Main server class.'''\n"
        "    def run(self):\n"
        "        return 'running'\n"
    )
    subprocess.run(git + ["-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Initial architecture with docs and server"], check=True)

    # 3. Commit 2: Add utility module
    utils_file = repo_dir / "utils.py"
    utils_file.write_text(
        "def compute_hash(val: str) -> str:\n"
        "    return val.strip().lower()\n"
    )
    subprocess.run(git + ["-C", str(repo_dir), "add", "utils.py"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Add compute_hash utility"], check=True)

    # 4. Commit 3: Add temporary buggy method
    app_file.write_text(
        "class WebServer:\n"
        "    '''Main server class.'''\n"
        "    def run(self):\n"
        "        return 'running'\n"
        "    def buggy_endpoint(self):\n"
        "        raise RuntimeError('broken')\n"
    )
    subprocess.run(git + ["-C", str(repo_dir), "add", "app.py"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Add buggy endpoint to server"], check=True)

    # 5. Commit 4: Revert Commit 3
    subprocess.run(git + ["-C", str(repo_dir), "revert", "--no-edit", "HEAD"], check=True)

    # 6. Commit 5: Rename utils.py to tools.py
    subprocess.run(git + ["-C", str(repo_dir), "mv", "utils.py", "tools.py"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Rename utils to tools"], check=True)

    # 7. Commit 6: Add and then delete temp file
    temp_file = repo_dir / "temp_scratch.txt"
    temp_file.write_text("temporary scratch notes\n")
    subprocess.run(git + ["-C", str(repo_dir), "add", "temp_scratch.txt"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Add temporary scratch file"], check=True)

    subprocess.run(git + ["-C", str(repo_dir), "rm", "temp_scratch.txt"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Remove temporary scratch file"], check=True)

    # 8. Add untracked secrets.json (must NOT be chunked or indexed)
    secret_file = repo_dir / "secrets.json"
    secret_file.write_text('{"api_key": "super_secret_untracked_token_12345"}')

    return repo_dir

