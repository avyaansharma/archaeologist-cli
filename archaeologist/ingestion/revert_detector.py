import re
from typing import Optional
from sqlmodel import Session, select
from archaeologist.storage.models import Commit

REVERT_MSG_PATTERN = re.compile(r'^Revert(?:\s+|:\s+)["\']?([^"\']+)["\']?', re.IGNORECASE)
REVERT_SHA_PATTERN = re.compile(r'(?:This reverts commit|revert commit|reverting commit)\s+([0-9a-fA-F]{7,40})', re.IGNORECASE)

def _escape_like(s: str) -> str:
    """Escapes SQL LIKE wildcard characters (%, _, \\)."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

def extract_revert_sha_from_message(message: str) -> Optional[str]:
    """Extracts authoritative SHA from git's 'This reverts commit <sha>' trailer if present."""
    if not message:
        return None
    m = REVERT_SHA_PATTERN.search(message)
    if m:
        return m.group(1).lower()
    return None

def detect_revert_from_message(message: str) -> Optional[str]:
    """Returns the subject line of the reverted commit if this looks like a git-generated revert."""
    if not message:
        return None
    first_line = message.strip().splitlines()[0].strip()
    
    # 1. Git default format: Revert "<subject>"
    m = re.match(r'^Revert\s+"(.*)"$', first_line, re.IGNORECASE)
    if m:
        return m.group(1).strip()
        
    # 2. Single quotes format: Revert '<subject>'
    m = re.match(r"^Revert\s+'(.*)'$", first_line, re.IGNORECASE)
    if m:
        return m.group(1).strip()
        
    # 3. Colon format: Revert: <subject>
    m = re.match(r'^Revert:\s*(.+)$', first_line, re.IGNORECASE)
    if m:
        subj = m.group(1).strip()
        if (subj.startswith('"') and subj.endswith('"')) or (subj.startswith("'") and subj.endswith("'")):
            subj = subj[1:-1].strip()
        return subj

    # 4. Git trailer check: if body contains "This reverts commit <sha>", message is definitely a revert
    sha_match = extract_revert_sha_from_message(message)
    if sha_match:
        # Return whatever subject is after Revert if present
        m = re.match(r'^Revert\s+(.+)$', first_line, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return None

def find_reverted_commit_by_sha(
    db_session: Session,
    target_sha: str,
    repo_id: Optional[str] = None
) -> Optional[Commit]:
    """Looks up original commit directly by SHA prefix (authoritative match)."""
    clean_sha = target_sha.lower()
    stmt = select(Commit).where(Commit.sha.like(f"{clean_sha}%"))
    if repo_id:
        stmt = stmt.where(Commit.repo_id == repo_id)
    return db_session.exec(stmt).first()

def find_reverted_commit(
    db_session: Session,
    reverted_subject: str,
    before_date,
    repo_id: Optional[str] = None
) -> Optional[Commit]:
    """git revert messages embed the ORIGINAL subject line, not its sha. Look it up by matching
    subject text among commits before this one. If multiple matches, take the most recent."""
    escaped_subject = _escape_like(reverted_subject)
    stmt = (select(Commit)
            .where(Commit.message.like(f"{escaped_subject}%", escape="\\"))
            .where(Commit.authored_date < before_date))
    if repo_id:
        stmt = stmt.where(Commit.repo_id == repo_id)
    stmt = stmt.order_by(Commit.authored_date.desc())
    return db_session.exec(stmt).first()
