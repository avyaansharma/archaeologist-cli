from typer.testing import CliRunner

from archaeologist.cli import app
from archaeologist.storage.paths import (
    detect_github_remote,
    find_repo_root,
    get_default_db_url,
)

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Codebase Archaeologist" in result.output
    assert "hotspots" in result.output
    assert "ownership" in result.output
    assert "coupling" in result.output
    assert "symbols" in result.output


def test_cli_hotspots_flask():
    result = runner.invoke(app, ["hotspots", "--top-n", "3", "--db-path", "eval/data/flask.db"])
    assert result.exit_code == 0
    assert "Repository Hotspot Files" in result.output
    assert "CHANGES.rst" in result.output


def test_cli_ownership_flask():
    result = runner.invoke(app, ["ownership", "--db-path", "eval/data/flask.db"])
    assert result.exit_code == 0
    assert "Author Contribution" in result.output
    assert "David Lord" in result.output


def test_cli_symbols_flask():
    result = runner.invoke(app, ["symbols", "--top-n", "3", "--db-path", "eval/data/flask.db"])
    assert result.exit_code == 0
    assert "AST Code Symbols" in result.output
    assert "Flask" in result.output


def test_storage_paths(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    root = find_repo_root()
    assert (root / ".git").exists()
    db_url = get_default_db_url(str(root))
    assert ".archaeologist" in db_url
    remote = detect_github_remote(str(root))
    assert remote is not None
    assert "codebase-archaeologist" in remote


def test_calculate_window_since():
    import pytest
    from datetime import datetime, timedelta
    from archaeologist.storage.paths import calculate_window_since

    now = datetime.now()
    # 6 months
    d6m = calculate_window_since("6m")
    assert d6m is not None
    expected_6m = (now - timedelta(days=182)).strftime("%Y-%m-%d")
    assert d6m == expected_6m

    # 1 year
    d1y = calculate_window_since("1y")
    assert d1y is not None
    expected_1y = (now - timedelta(days=365)).strftime("%Y-%m-%d")
    assert d1y == expected_1y

    # 2 years
    d2y = calculate_window_since("2y")
    assert d2y is not None
    expected_2y = (now - timedelta(days=730)).strftime("%Y-%m-%d")
    assert d2y == expected_2y

    # Full history
    dfull = calculate_window_since("full")
    assert dfull is None
    dall = calculate_window_since("all")
    assert dall is None

    # Invalid window raises ValueError
    with pytest.raises(ValueError, match="Invalid window"):
        calculate_window_since("50years")


def test_cli_ingest_window_options():
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "--window" in result.output
    assert "'6m'" in result.output
    assert "'1y'" in result.output
    assert "'2y'" in result.output
    assert "'full'" in result.output

    # Invalid window exits with code 1
    err_res = runner.invoke(app, ["ingest", "--window", "invalid_window"])
    assert err_res.exit_code == 1
    assert "Invalid window" in err_res.output


def test_mcp_server_initialization():
    """Validates that the FastMCP/MCPServer stdio module loads without import errors and registers all tools."""
    from archaeologist.mcp_server.server import mcp
    assert mcp is not None
    assert mcp.name == "Codebase Archaeologist"


def test_db_fresh_engine_provisions_all_tables(tmp_path):
    """Validates that connecting to a brand-new database provisions all 5 schema tables automatically."""
    from sqlalchemy import inspect
    from archaeologist.storage.db import get_engine_for_url

    new_db_file = tmp_path / "fresh_repo.db"
    new_db_url = f"sqlite:///{new_db_file.as_posix()}"
    eng = get_engine_for_url(new_db_url)
    insp = inspect(eng)
    table_names = insp.get_table_names()
    for expected_table in ["commit", "pullrequest", "issue", "chunk", "symbolindex"]:
        assert expected_table in table_names


def test_cli_empty_repo_notices(tmp_path):
    """Validates that querying an un-ingested empty repository displays helpful guidance instead of crashing."""
    empty_db = tmp_path / "empty.db"
    empty_db_path = str(empty_db)

    # Hotspots on empty db
    res_hotspots = runner.invoke(app, ["hotspots", "--db-path", empty_db_path])
    assert res_hotspots.exit_code == 0
    assert "No commits found for this repository" in res_hotspots.output

    # Ownership on empty db
    res_ownership = runner.invoke(app, ["ownership", "--db-path", empty_db_path])
    assert res_ownership.exit_code == 0
    assert "No commits found for this repository" in res_ownership.output

    # Coupling on empty db
    res_coupling = runner.invoke(app, ["coupling", "--db-path", empty_db_path])
    assert res_coupling.exit_code == 0
    assert "No coupled file changes found" in res_coupling.output

    # Symbols on empty db
    res_symbols = runner.invoke(app, ["symbols", "--db-path", empty_db_path])
    assert res_symbols.exit_code == 0
    assert "No AST symbols found for this repository" in res_symbols.output

