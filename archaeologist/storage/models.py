from datetime import datetime
from typing import Optional, List, Any
from sqlmodel import SQLModel, Field, JSON, Column, select

class Commit(SQLModel, table=True):
    sha: str = Field(primary_key=True)
    repo_id: Optional[str] = Field(default=None, index=True)
    author_name: str
    author_email: str
    authored_date: datetime
    message: str
    files_changed: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    symbols_modified: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    insertions: int = 0
    deletions: int = 0
    is_revert: bool = False
    reverts_sha: Optional[str] = None          # points to the commit this one reverts
    superseded_by_sha: Optional[str] = None    # set later if this commit is itself reverted
    diff_summary: Optional[str] = None         # LLM-generated, cached
    raw_diff_truncated: Optional[str] = None   # first N chars of diff, for small diffs store full

class RepoMeta(SQLModel, table=True):
    __tablename__ = "repo_meta"
    repo_id: str = Field(primary_key=True)
    repo_name: Optional[str] = None
    repo_url: Optional[str] = None
    embedder_provider: Optional[str] = None
    embedder_dimension: Optional[int] = None
    last_ingested_at: Optional[datetime] = None
    key: Optional[str] = None
    value: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class PullRequest(SQLModel, table=True):
    repo_id: str = Field(default="", primary_key=True, index=True)
    number: int = Field(primary_key=True)
    title: str
    body: Optional[str] = None
    state: str                                  # open|closed|merged
    author: str = "unknown"
    created_at: datetime
    merged_at: Optional[datetime] = None
    merge_commit_sha: Optional[str] = None
    linked_issue_numbers: List[int] = Field(default_factory=list, sa_column=Column(JSON))
    linked_commit_shas: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    review_comments: List[dict] = Field(default_factory=list, sa_column=Column(JSON))
    comments: List[dict] = Field(default_factory=list, sa_column=Column(JSON))

    def __init__(self, **data):
        if data.get("repo_id") is None:
            data["repo_id"] = ""
        super().__init__(**data)

class Issue(SQLModel, table=True):
    repo_id: str = Field(default="", primary_key=True, index=True)
    number: int = Field(primary_key=True)
    title: str
    body: Optional[str] = None
    state: str
    author: Optional[str] = "unknown"
    labels: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime
    closed_at: Optional[datetime] = None
    close_reason: Optional[str] = None          # "completed" | "not_planned" | None
    linked_pr_numbers: List[int] = Field(default_factory=list, sa_column=Column(JSON))
    linked_commit_shas: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    comments: List[dict] = Field(default_factory=list, sa_column=Column(JSON))

    def __init__(self, **data):
        if data.get("repo_id") is None:
            data["repo_id"] = ""
        super().__init__(**data)

class Chunk(SQLModel, table=True):
    id: str = Field(primary_key=True)            # deterministic uuid5
    repo_id: Optional[str] = Field(default=None, index=True)
    source_type: str                              # "commit"|"pr"|"issue"|"thread_summary"
    source_id: str                                 # sha, or "pr#123", "issue#45"
    text: str
    timestamp: datetime
    file_paths: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    symbols_modified: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    related_ids: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    is_reverted: bool = False
    token_count: int = 0
    embedded: bool = False                         # ingestion progress flag, for resumability

class SymbolIndex(SQLModel, table=True):
    repo_id: str = Field(default="", primary_key=True, index=True)
    symbol_id: str = Field(primary_key=True)      # e.g. "src/auth.py::AuthService::login"
    file_path: str
    symbol_name: str
    kind: str                                     # "class" | "function" | "method"
    commit_count: int = 0

    def __init__(self, **data):
        if data.get("repo_id") is None:
            data["repo_id"] = ""
        super().__init__(**data)

def is_valid_commit_sha(session, sha_str: str, repo_id: Optional[str] = None) -> bool:
    if not sha_str or len(sha_str) < 7:
        return False
    from archaeologist.utils.security import escape_like
    escaped = escape_like(sha_str)
    stmt = select(Commit.sha).where(Commit.sha.like(f"{escaped}%", escape="\\"))
    if repo_id:
        stmt = stmt.where(Commit.repo_id == repo_id)
    result = session.exec(stmt).first()
    return result is not None

def is_valid_pr_or_issue(session, ref_num: Any, repo_id: Optional[str] = None) -> bool:
    try:
        clean_num = int(ref_num)
    except (ValueError, TypeError):
        return False

    pr_stmt = select(PullRequest.number).where(PullRequest.number == clean_num)
    issue_stmt = select(Issue.number).where(Issue.number == clean_num)
    if repo_id:
        clean_repo = repo_id.strip().lower()
        base_repo = clean_repo.split("/")[-1]
        from archaeologist.utils.security import escape_like
        escaped_base = escape_like(base_repo)
        repo_filter_pr = (
            (PullRequest.repo_id == clean_repo) |
            (PullRequest.repo_id.like(f"%/{escaped_base}", escape="\\")) |
            (PullRequest.repo_id == base_repo)
        )
        repo_filter_iss = (
            (Issue.repo_id == clean_repo) |
            (Issue.repo_id.like(f"%/{escaped_base}", escape="\\")) |
            (Issue.repo_id == base_repo)
        )
        pr_stmt = pr_stmt.where(repo_filter_pr)
        issue_stmt = issue_stmt.where(repo_filter_iss)

    if session.exec(pr_stmt).first() is not None:
        return True
    return session.exec(issue_stmt).first() is not None

def _repo_filter(model_col, repo_id: Optional[str]):
    if not repo_id:
        return True
    from sqlalchemy import func
    clean_repo = repo_id.strip()
    clean_lower = clean_repo.lower()
    base_repo = clean_repo.split("/")[-1]
    base_lower = clean_lower.split("/")[-1]
    from archaeologist.utils.security import escape_like
    escaped_base = escape_like(base_repo)
    escaped_base_lower = escape_like(base_lower)
    return (
        (model_col == clean_repo) |
        (func.lower(model_col) == clean_lower) |
        (model_col.like(f"%/{escaped_base}", escape="\\")) |
        (func.lower(model_col).like(f"%/{escaped_base_lower}", escape="\\")) |
        (model_col == base_repo) |
        (func.lower(model_col) == base_lower)
    )

def find_candidate_symbols(session, text: str, repo_id: Optional[str] = None, limit: int = 8) -> List[dict]:
    """Finds candidate code symbols matching key tokens in the question text to ground planning."""
    if not text:
        return []
    import re
    from archaeologist.utils.security import escape_like
    
    words = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]{3,}', text))
    stopwords = {"what", "when", "where", "which", "does", "from", "with", "this", "that", "have", "been", "were", "commit", "commits", "pull", "request", "issue", "repository", "history", "code", "file", "method", "function", "class", "architecture", "handle", "using"}
    candidates = [w for w in words if w.lower() not in stopwords]
    
    if not candidates:
        return []
    
    matched = []
    seen = set()
    for word in candidates:
        escaped = escape_like(word)
        stmt = select(SymbolIndex).where(
            (SymbolIndex.symbol_name.like(f"%{escaped}%", escape="\\")) |
            (SymbolIndex.file_path.like(f"%{escaped}%", escape="\\"))
        )
        if repo_id:
            stmt = stmt.where(_repo_filter(SymbolIndex.repo_id, repo_id))
        stmt = stmt.limit(limit)
        results = session.exec(stmt).all()
        for r in results:
            if r.symbol_id not in seen:
                matched.append({
                    "symbol_name": r.symbol_name,
                    "file_path": r.file_path,
                    "kind": r.kind
                })
                seen.add(r.symbol_id)
                if len(matched) >= limit:
                    return matched
    return matched

def find_candidate_forensics(session, text: str, repo_id: Optional[str] = None) -> List[str]:
    """Extracts candidate PRs, Issues, Commit SHAs, and AST symbols matching query text."""
    if not text:
        return []
    import re
    from archaeologist.utils.security import escape_like
    
    evidence = []
    # 1. PR / Issue numbers (e.g. #452, PR #494, Issue #486)
    pr_issue_nums = re.findall(r'#(\d+)', text)
    for num_str in pr_issue_nums[:4]:
        try:
            num = int(num_str)
            pr_stmt = select(PullRequest).where(PullRequest.number == num)
            if repo_id:
                pr_stmt = pr_stmt.where(_repo_filter(PullRequest.repo_id, repo_id))
            pr = session.exec(pr_stmt).first()
            if pr:
                evidence.append(f"Pull Request #{pr.number}: {pr.title}")
            iss_stmt = select(Issue).where(Issue.number == num)
            if repo_id:
                iss_stmt = iss_stmt.where(_repo_filter(Issue.repo_id, repo_id))
            issue = session.exec(iss_stmt).first()
            if issue and (not pr or pr.title != issue.title):
                evidence.append(f"Issue #{issue.number}: {issue.title}")
        except Exception:
            pass

    # 2. Short / full hex SHAs (e.g. 06dc845, 5e5f3ee, 2d24115)
    shas = re.findall(r'\b[0-9a-fA-F]{7,40}\b', text)
    for sha in shas[:3]:
        escaped_sha = escape_like(sha)
        stmt = select(Commit).where(Commit.sha.like(f"{escaped_sha}%", escape="\\")).limit(1)
        if repo_id:
            stmt = stmt.where(_repo_filter(Commit.repo_id, repo_id))
        c = session.exec(stmt).first()
        if c:
            msg_line = c.message.splitlines()[0] if c.message else ""
            evidence.append(f"Commit {c.sha[:7]}: {msg_line}")

    # 3. Symbols
    symbols = find_candidate_symbols(session, text, repo_id=repo_id, limit=6)
    for s in symbols:
        evidence.append(f"Symbol: {s['symbol_name']} ({s['kind']} in {s['file_path']})")

    return evidence




