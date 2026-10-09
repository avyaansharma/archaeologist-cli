import pytest
from datetime import datetime, timedelta
from archaeologist.storage.paths import calculate_window_since
from archaeologist.ingestion.git_parser import iter_commits


def test_calculate_window_since_exact_cutoffs():
    now = datetime.now()
    
    # 6 months (182 days)
    w_6m = calculate_window_since("6m")
    assert w_6m == (now - timedelta(days=182)).strftime("%Y-%m-%d")
    assert calculate_window_since("6months") == w_6m
    assert calculate_window_since("6-month") == w_6m

    # 1 year (365 days)
    w_1y = calculate_window_since("1y")
    assert w_1y == (now - timedelta(days=365)).strftime("%Y-%m-%d")
    assert calculate_window_since("1year") == w_1y

    # 2 years (730 days)
    w_2y = calculate_window_since("2y")
    assert w_2y == (now - timedelta(days=730)).strftime("%Y-%m-%d")
    assert calculate_window_since("2years") == w_2y

    # Full history
    assert calculate_window_since("full") is None
    assert calculate_window_since("all") is None
    assert calculate_window_since(None) is None

    # Invalid options
    with pytest.raises(ValueError, match="Invalid window '3m'"):
        calculate_window_since("3m")
    with pytest.raises(ValueError, match="Invalid window '5y'"):
        calculate_window_since("5y")


def test_git_parser_window_filtering_accuracy():
    """Verify that git commit parsing accurately respects the cutoff date."""
    import os
    repo_path = "scratch/mega-mind-skills"
    if not os.path.exists(repo_path):
        pytest.skip("scratch/mega-mind-skills not available for live git test")

    cutoff_6m = calculate_window_since("6m")
    commits_6m = list(iter_commits(repo_path, since=cutoff_6m))
    commits_full = list(iter_commits(repo_path, since=None))

    assert len(commits_full) == 50
    # Commits strictly on or after 6m cutoff
    assert len(commits_6m) <= len(commits_full)
    for c in commits_6m:
        assert c["authored_date"].strftime("%Y-%m-%d") >= cutoff_6m
