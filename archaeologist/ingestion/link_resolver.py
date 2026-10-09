import re
from typing import List, Set, Tuple, Any, Optional


ISSUE_REF = re.compile(r'#(\d+)')
CLOSES_CLAUSE = re.compile(
    r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)(?::\s*|\s+)((?:#\d+(?:[,\s]+and\s*|[,\s]+)*)+)',
    re.IGNORECASE
)
# Match github.com/owner/repo/issues/123 or pull/123
FULL_URL_REF = re.compile(r'github\.com/[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+/(?:issues|pull)/(\d+)')

# Require at least one non-decimal hex character (a-fA-F) (§7 Fix)
SHA_REF = re.compile(r'\b(?=[a-fA-F0-9]{7,40}\b)(?=\w*[a-fA-F])[a-fA-F0-9]{7,40}\b')

def extract_refs(text: str) -> dict:
    """Returns {'closes': [ids], 'mentions': [ids], 'shas': [shas]}."""
    if not text:
        return {"closes": [], "mentions": [], "shas": []}
    closes = set()
    for clause in CLOSES_CLAUSE.findall(text):
        for num_str in ISSUE_REF.findall(clause):
            closes.add(int(num_str))
    all_refs = {int(n) for n in ISSUE_REF.findall(text)} | {int(n) for n in FULL_URL_REF.findall(text)}
    mentions = all_refs - closes
    shas = {s.lower() for s in SHA_REF.findall(text) if len(s) >= 7}
    
    return {
        "closes": sorted(list(closes)),
        "mentions": sorted(list(mentions)),
        "shas": sorted(list(shas))
    }

def update_cross_links(
    session_or_prs: Any, 
    issue_list: Optional[List[dict]] = None,
    repo_id: Optional[str] = None
) -> Any:
    """Updates linked_issue_numbers on PRs and linked_pr_numbers on issues in-place, plus linked_commit_shas.
    Supports either an ORM Session: update_cross_links(session, repo_id=repo_id)
    or dict lists: update_cross_links(pr_list, issue_list)
    """
    if issue_list is None and hasattr(session_or_prs, "exec"):
        session = session_or_prs
        from sqlmodel import select
        from archaeologist.storage.models import PullRequest, Issue

        pr_query = select(PullRequest)
        iss_query = select(Issue)
        if repo_id:
            pr_query = pr_query.where(PullRequest.repo_id == repo_id)
            iss_query = iss_query.where(Issue.repo_id == repo_id)

        all_prs = session.exec(pr_query).all()
        all_issues = session.exec(iss_query).all()

        pr_map = {(p.repo_id or "", p.number): p for p in all_prs}
        issue_map = {(i.repo_id or "", i.number): i for i in all_issues}

        for p in all_prs:
            p_repo = p.repo_id or ""
            linked_issues = set(p.linked_issue_numbers or [])
            linked_shas = set(p.linked_commit_shas or [])

            all_text = f"{p.title or ''} {p.body or ''} "
            for c in (p.review_comments or []) + (p.comments or []):
                if isinstance(c, dict):
                    all_text += f"{c.get('body', '')} "
                elif isinstance(c, str):
                    all_text += f"{c} "

            refs = extract_refs(all_text)
            linked_shas.update(refs["shas"])

            # Only link issues that exist within the SAME repository
            for inum in refs["closes"] + refs["mentions"]:
                issue_key = (p_repo, inum)
                if issue_key in issue_map:
                    linked_issues.add(inum)
                    target_issue = issue_map[issue_key]
                    existing_prs = set(target_issue.linked_pr_numbers or [])
                    existing_prs.add(p.number)
                    target_issue.linked_pr_numbers = sorted(list(existing_prs))
                    session.add(target_issue)

            p.linked_issue_numbers = sorted(list(linked_issues))
            p.linked_commit_shas = sorted(list(linked_shas))
            session.add(p)

        for i in all_issues:
            i_repo = i.repo_id or ""
            linked_prs = set(i.linked_pr_numbers or [])
            linked_shas = set(i.linked_commit_shas or [])

            all_text = f"{i.title or ''} {i.body or ''} "
            for c in i.comments or []:
                if isinstance(c, dict):
                    all_text += f"{c.get('body', '')} "
                elif isinstance(c, str):
                    all_text += f"{c} "

            refs = extract_refs(all_text)
            linked_shas.update(refs["shas"])
            for pr_num in refs["closes"] + refs["mentions"]:
                pr_key = (i_repo, pr_num)
                if pr_key in pr_map:
                    linked_prs.add(pr_num)
                    target_pr = pr_map[pr_key]
                    existing_issues = set(target_pr.linked_issue_numbers or [])
                    existing_issues.add(i.number)
                    target_pr.linked_issue_numbers = sorted(list(existing_issues))
                    session.add(target_pr)

            i.linked_pr_numbers = sorted(list(linked_prs))
            i.linked_commit_shas = sorted(list(linked_shas))
            session.add(i)

        session.commit()
        return all_prs, all_issues

    # Fallback for dict lists
    pr_list = session_or_prs
    if issue_list is None:
        issue_list = []

    issue_map = {(i.get("repo_id") or "", i["number"]): i for i in issue_list}
    pr_map = {(p.get("repo_id") or "", p["number"]): p for p in pr_list}

    for p in pr_list:
        if "linked_issue_numbers" not in p:
            p["linked_issue_numbers"] = []
        if "linked_commit_shas" not in p:
            p["linked_commit_shas"] = []

        all_text = f"{p.get('title', '')} {p.get('body', '')} "
        comments_list = p.get("review_comments", []) + p.get("comments", [])
        for c in comments_list:
            if isinstance(c, dict):
                all_text += f"{c.get('body', '')} "
            elif isinstance(c, str):
                all_text += f"{c} "

        refs = extract_refs(all_text)
        all_issues = set(p["linked_issue_numbers"]) | set(refs["closes"]) | set(refs["mentions"])
        p["linked_issue_numbers"] = sorted(list(all_issues))
        p["linked_commit_shas"] = sorted(list(set(p["linked_commit_shas"]) | set(refs["shas"])))

        # Update back-references on issues
        p_repo = p.get("repo_id") or ""
        for issue_num in p["linked_issue_numbers"]:
            issue_key = (p_repo, issue_num)
            if issue_key in issue_map:
                target_issue = issue_map[issue_key]
                if "linked_pr_numbers" not in target_issue:
                    target_issue["linked_pr_numbers"] = []
                if p["number"] not in target_issue["linked_pr_numbers"]:
                    target_issue["linked_pr_numbers"].append(p["number"])
                    target_issue["linked_pr_numbers"].sort()

    for i in issue_list:
        if "linked_pr_numbers" not in i:
            i["linked_pr_numbers"] = []
        if "linked_commit_shas" not in i:
            i["linked_commit_shas"] = []

        all_text = f"{i.get('title', '')} {i.get('body', '')} "
        for c in i.get("comments", []):
            if isinstance(c, dict):
                all_text += f"{c.get('body', '')} "
            elif isinstance(c, str):
                all_text += f"{c} "

        refs = extract_refs(all_text)
        i["linked_commit_shas"] = sorted(list(set(i["linked_commit_shas"]) | set(refs["shas"])))

        i_repo = i.get("repo_id") or ""
        for pr_num in refs["closes"] + refs["mentions"]:
            pr_key = (i_repo, pr_num)
            if pr_key in pr_map:
                if pr_num not in i["linked_pr_numbers"]:
                    i["linked_pr_numbers"].append(pr_num)
                    i["linked_pr_numbers"].sort()
                target_pr = pr_map[pr_key]
                if "linked_issue_numbers" not in target_pr:
                    target_pr["linked_issue_numbers"] = []
                if i["number"] not in target_pr["linked_issue_numbers"]:
                    target_pr["linked_issue_numbers"].append(i["number"])
                    target_pr["linked_issue_numbers"].sort()

    return pr_list, issue_list

