import pytest
import os
from archaeologist.utils.security import validate_repo_path, sanitize_file_path, sanitize_sha

def test_validate_repo_path_success(tmp_path):
    repo_dir = tmp_path / "my_repo"
    repo_dir.mkdir()
    validated = validate_repo_path(str(repo_dir))
    assert os.path.exists(validated)

def test_validate_repo_path_invalid():
    with pytest.raises(ValueError):
        validate_repo_path("/non/existent/path/that/should/fail")

def test_sanitize_file_path_normal(tmp_path):
    base_dir = tmp_path / "repo"
    base_dir.mkdir()
    sub_file = base_dir / "src" / "auth.py"
    sub_file.parent.mkdir()
    sub_file.touch()

    rel_path = sanitize_file_path(str(base_dir), "src/auth.py")
    assert rel_path == "src/auth.py"

def test_sanitize_file_path_traversal_attack(tmp_path):
    base_dir = tmp_path / "repo"
    base_dir.mkdir()
    with pytest.raises(ValueError):
        sanitize_file_path(str(base_dir), "../../etc/passwd")

def test_sanitize_file_path_sibling_dir_attack(tmp_path):
    base_dir = tmp_path / "repo"
    base_dir.mkdir()
    sibling_dir = tmp_path / "repo-evil-sibling"
    sibling_dir.mkdir()
    with pytest.raises(ValueError):
        sanitize_file_path(str(base_dir), "../repo-evil-sibling/secret.py")

def test_sanitize_sha():
    valid_sha = "a1b2c3d4e5f60789"
    assert sanitize_sha(valid_sha) == valid_sha

    with pytest.raises(ValueError):
        sanitize_sha("a1b2c3d; rm -rf /")
