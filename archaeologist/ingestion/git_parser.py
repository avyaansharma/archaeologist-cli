import os
import subprocess
from datetime import datetime
from typing import Iterator, Optional
from archaeologist.utils.security import validate_repo_path, sanitize_sha

COMMIT_DELIM = "\x1e"   # record separator
FIELD_DELIM = "\x1f"    # unit separator

def count_commits(repo_path: str, since: Optional[str] = None) -> int:
    """Quickly returns total commit count using git rev-list for accurate progress calculation."""
    try:
        validated_path = validate_repo_path(repo_path)
        cmd = ["git", "-C", validated_path, "rev-list", "--count", "HEAD"]
        if since:
            cmd += [f"--since={since}"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if res.returncode == 0 and res.stdout.strip().isdigit():
            return int(res.stdout.strip())
    except Exception:
        pass
    return 0

def iter_commits(repo_path: str, since: Optional[str] = None) -> Iterator[dict]:
    """Yields raw commit dicts one at a time. O(1) memory regardless of repo size."""
    validated_path = validate_repo_path(repo_path)
    
    # Prefix format with COMMIT_DELIM and end with FIELD_DELIM + "NUMSTAT"
    fmt = COMMIT_DELIM + FIELD_DELIM.join(["%H", "%an", "%ae", "%aI", "%s%n%b"]) + FIELD_DELIM + "NUMSTAT"
    cmd = ["git", "-C", validated_path, "log", f"--pretty=format:{fmt}", "--numstat"]
    if since:
        cmd += [f"--since={since}"]
    
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
        return p
    if "{" in p and "}" in p:
        import re
        return re.sub(r'\{[^{}]* => ([^{}]*)\}', r'\1', p).strip().strip('"\'')
    return p.split(" => ")[-1].strip().strip('"\'')

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

def get_commit_diff(repo_path: str, sha: str, max_bytes: int = 20_000) -> str:
    """Returns the diff text unified format for a commit, limited to max_bytes."""
    validated_path = validate_repo_path(repo_path)
    clean_sha = sanitize_sha(sha)
    cmd = ["git", "-C", validated_path, "show", clean_sha, "--format=", "--unified=3"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    diff = result.stdout
    return diff[:max_bytes]

def is_merge_commit(repo_path: str, sha: str) -> bool:
    """Checks if a commit is a merge commit (has 2 or more parent commits)."""
    validated_path = validate_repo_path(repo_path)
    clean_sha = sanitize_sha(sha)
    cmd = ["git", "-C", validated_path, "rev-list", "--parents", "-n", "1", clean_sha]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    parents = result.stdout.strip().split()
    return len(parents) > 2
