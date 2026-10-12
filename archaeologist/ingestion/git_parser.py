import subprocess
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from archaeologist.storage.paths import find_repo_root
from archaeologist.utils.security import sanitize_sha, validate_repo_path

COMMIT_DELIM = "\x1e"   # record separator
FIELD_DELIM = "\x1f"    # unit separator

def _get_monorepo_subpath(validated_path: str) -> str | None:
    """Returns the relative subdirectory path if validated_path is inside a git monorepo root."""
    try:
        top_level = find_repo_root(validated_path)
        val_abs = Path(validated_path).resolve()
        if top_level != val_abs and top_level in val_abs.parents:
            rel = val_abs.relative_to(top_level).as_posix()
            if rel and rel != ".":
                return rel
    except Exception:
        pass
    return None

def count_commits(repo_path: str, since: str | None = None) -> int:
    """Quickly returns total commit count using git rev-list for accurate progress calculation."""
    try:
        validated_path = validate_repo_path(repo_path)
        cmd = ["git", "-c", "core.quotepath=false", "-C", validated_path, "rev-list", "--count", "HEAD"]
        if since:
            cmd += [f"--since={since}"]
        subpath = _get_monorepo_subpath(validated_path)
        if subpath:
            cmd += ["--", subpath]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0 and res.stdout.strip().isdigit():
            return int(res.stdout.strip())
    except Exception:
        pass
    return 0

def iter_commits(repo_path: str, since: str | None = None) -> Iterator[dict]:
    """Yields raw commit dicts one at a time. O(1) memory regardless of repo size."""
    validated_path = validate_repo_path(repo_path)

    # Prefix format with COMMIT_DELIM and end with FIELD_DELIM + "NUMSTAT"
    fmt = COMMIT_DELIM + FIELD_DELIM.join(["%H", "%an", "%ae", "%aI", "%s%n%b"]) + FIELD_DELIM + "NUMSTAT"
    cmd = ["git", "-c", "core.quotepath=false", "-C", validated_path, "log", f"--pretty=format:{fmt}", "--numstat"]
    if since:
        cmd += [f"--since={since}"]
    subpath = _get_monorepo_subpath(validated_path)
    if subpath:
        cmd += ["--", subpath]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    buffer = ""
    try:
        for line in proc.stdout:
            buffer += line
            if COMMIT_DELIM in buffer:
                records = buffer.split(COMMIT_DELIM)
                buffer = COMMIT_DELIM + records[-1]
                for record in records[:-1]:
                    if record.strip():
                        yield _parse_commit_record(record)
        if buffer.strip():
            for record in buffer.split(COMMIT_DELIM):
                if record.strip():
                    yield _parse_commit_record(record)
    finally:
        proc.terminate()
        proc.wait()

def _normalize_numstat_path(path: str) -> str:
    """Normalizes git numstat paths, extracting the target path from rename expressions like 'src/{old => new}.py'."""
    p = path.strip().strip('"\'')
    if " => " not in p:
        return p.lstrip("/\\")
    if "{" in p and "}" in p:
        import re
        norm = re.sub(r'\{[^{}]* => ([^{}]*)\}', r'\1', p)
        norm = norm.replace("//", "/").strip().strip('"\'')
        return norm.lstrip("/\\")
    return p.split(" => ")[-1].strip().strip('"\'').lstrip("/\\")

def _parse_commit_record(record: str) -> dict:
    clean_record = record.lstrip(COMMIT_DELIM).strip()

    # Separate the metadata fields from the numstat block
    part1, _, part2 = clean_record.partition(FIELD_DELIM + "NUMSTAT")

    # Split metadata fields: [sha, author, email, date, message]
    fields = part1.split(FIELD_DELIM, 4)

    sha = fields[0].strip() if len(fields) > 0 else "unknown"
    author = fields[1].strip() if len(fields) > 1 else "unknown"
    email = fields[2].strip() if len(fields) > 2 else "unknown"
    date_iso = fields[3].strip() if len(fields) > 3 else datetime.utcnow().isoformat()
    message = fields[4].strip() if len(fields) > 4 else ""

    files_changed = []
    insertions = 0
    deletions = 0

    # Parse numstat lines
    for line in part2.strip().splitlines():
        parts_num = line.split("\t")
        if len(parts_num) == 3:
            ins, dels, raw_path = parts_num
            path = _normalize_numstat_path(raw_path)
            files_changed.append(path)
            insertions += int(ins) if ins.isdigit() else 0
            deletions += int(dels) if dels.isdigit() else 0

    try:
        dt = datetime.fromisoformat(date_iso)
    except Exception:
        dt = datetime.utcnow()

    return {
        "sha": sha,
        "author_name": author,
        "author_email": email,
        "authored_date": dt,
        "message": message,
        "files_changed": files_changed,
        "insertions": insertions,
        "deletions": deletions,
    }

def is_merge_commit(repo_path: str, sha: str) -> bool:
    """Checks if a commit is a merge commit (has 2 or more parent commits)."""
    validated_path = validate_repo_path(repo_path)
    clean_sha = sanitize_sha(sha)
    cmd = ["git", "-c", "core.quotepath=false", "-C", validated_path, "rev-list", "--parents", "-n", "1", clean_sha]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    parents = result.stdout.strip().split()
    return len(parents) > 2

def get_commit_diff(repo_path: str, sha: str, max_bytes: int = 20_000) -> str:
    """Returns the diff text unified format for a commit, limited to max_bytes without buffering full diff."""
    validated_path = validate_repo_path(repo_path)
    clean_sha = sanitize_sha(sha)

    cmd = ["git", "-c", "core.quotepath=false", "-C", validated_path, "show"]
    if is_merge_commit(validated_path, clean_sha):
        cmd.extend(["-m", "--first-parent"])
    cmd.extend([clean_sha, "--format=", "--unified=3"])

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace")
    diff_lines = []
    total_bytes = 0
    try:
        if proc.stdout:
            for line in proc.stdout:
                diff_lines.append(line)
                total_bytes += len(line.encode("utf-8", errors="replace"))
                if total_bytes >= max_bytes:
                    break
    finally:
        proc.terminate()
        proc.wait()

    return "".join(diff_lines)[:max_bytes]
