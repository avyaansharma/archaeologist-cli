import re
from typing import Optional
from sqlmodel import Session, select
from archaeologist.storage.models import Commit

REVERT_MSG_PATTERN = re.compile(r'^Revert(?:\s+|:\s+)["\']?([^"\']+)["\']?', re.IGNORECASE)

def _escape_like(s: str) -> str:
    """Escapes SQL LIKE wildcard characters (%, _, \\)."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

def detect_revert_from_message(message: str) -> Optional[str]:
    """Returns the subject line of the reverted commit if this looks like a git-generated revert."""
    if not message:
        return None
    first_line = message.strip().splitlines()[0].strip()
    
    # Git default format: Revert "<subject>"
    m = re.match(r'^Revert(?:\s+|:\s+)"(.*)"$', first_line, re.IGNORECASE)
    if m:
        return m.group(1).strip()
        
    # Single quotes format: Revert '<subject>'
    m = re.match(r"^Revert(?:\s+|:\s+)'(.*)'$", first_line, re.IGNORECASE)
    if m:
        return m.group(1).strip()
        
    # Unquoted format: Revert: <subject> or Revert <subject>
    m = re.match(r'^Revert(?:\s+|:\s+)(.+)$', first_line, re.IGNORECASE)
    if m:
        subj = m.group(1).strip()
        if (subj.startswith('"') and subj.endswith('"')) or (subj.startswith("'") and subj.endswith("'")):
            subj = subj[1:-1].strip()
        return subj
    return None

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
