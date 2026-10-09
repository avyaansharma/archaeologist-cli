import os
import sys
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from github import Github, Auth
from dotenv import load_dotenv

load_dotenv()

class GitHubIngestionClient:
    def __init__(self, repo_url: str, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        if self.token:
            try:
                auth = Auth.Token(self.token)
                self.gh = Github(auth=auth, timeout=10, retry=0)
            except Exception:
                self.gh = Github(self.token, timeout=10, retry=0)
        else:
            self.gh = Github(timeout=10, retry=0)
        
        clean_url = repo_url.rstrip("/").removesuffix(".git")
        match = re.search(r'github\.com/([^/]+)/([^/]+)', clean_url)
        if not match:
            raise ValueError(f"Invalid GitHub URL: '{repo_url}'")
            
        self.owner = match.group(1)
        self.repo_name = match.group(2)
        try:
            self.repo = self.gh.get_repo(f"{self.owner}/{self.repo_name}")
        except Exception as e:
            err_str = str(e)
            if "401" in err_str or "Bad credentials" in err_str:
                print("Notice: GITHUB_TOKEN is invalid or expired (401 Bad credentials). Retrying with public unauthenticated GitHub access...", file=sys.stderr)
                try:
                    self.gh = Github(timeout=10, retry=0)
                    self.repo = self.gh.get_repo(f"{self.owner}/{self.repo_name}")
                    self.token = None
                except Exception as retry_e:
                    print(f"Notice: Public GitHub API fallback failed ({retry_e}). Continuing in offline mode with local git history.", file=sys.stderr)
                    self.repo = None
            elif "404" in err_str:
                print(f"Notice: GitHub repository '{self.owner}/{self.repo_name}' not found or private. Continuing in offline mode with local git history.", file=sys.stderr)
                self.repo = None
            elif "rate limit" in err_str.lower() or "403" in err_str:
                print("Notice: GitHub API rate limit reached. Continuing in offline mode with local git history.", file=sys.stderr)
                self.repo = None
            else:
                print(f"Notice: Could not access GitHub repository metadata for '{self.owner}/{self.repo_name}': {e}", file=sys.stderr)
                self.repo = None


    def fetch_pull_requests(
        self,
        state: str = "all",
        limit: int = 500,
        sort: str = "updated",
        direction: str = "desc",
        since: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Fetches PRs sorted by updated (descending) with genuine inline review comments & issue comments."""
        if not self.repo:
            return []

        prs_data = []
        try:
            rate_obj = getattr(self.gh.get_rate_limit(), 'core', None) or getattr(self.gh.get_rate_limit(), 'rate', None)
            rem = getattr(rate_obj, 'remaining', 0) if rate_obj else 0

            if rem < 5:
                print(f"Notice: GitHub API rate limit low ({rem} remaining). Skipping detailed GitHub REST API fetching.", file=sys.stderr)
                return []

            prs = self.repo.get_pulls(state=state, sort=sort, direction=direction)
            count = 0
            skip_comments = not bool(self.token)
            for pr in prs:
                if count >= limit:
                    break
                
                if since and pr.updated_at:
                    pr_up = pr.updated_at.replace(tzinfo=None) if hasattr(pr.updated_at, 'tzinfo') and pr.updated_at.tzinfo else pr.updated_at
                    since_cmp = since.replace(tzinfo=None) if hasattr(since, 'tzinfo') and since.tzinfo else since
                    if pr_up < since_cmp:
                        break

                body_text = pr.body or ""
                linked_issues = [int(n) for n in re.findall(r'(?:fixes|resolves|closes|refs)\s+#(\d+)', body_text, re.IGNORECASE)]

                
                # 1. Issue-level discussion comments
                comments = []
                if not skip_comments:
                    try:
                        for c in pr.get_issue_comments():
                            comments.append({
                                "author": c.user.login if c.user else "unknown",
                                "user": c.user.login if c.user else "unknown",
                                "body": c.body or "",
                                "created_at": c.created_at.isoformat()
                            })
                    except Exception as e:
                        if "rate limit" in str(e).lower() or "403" in str(e) or "429" in str(e):
                            print("Warning: GitHub API rate limit hit while fetching PR issue comments. Continuing with PR metadata only.", file=sys.stderr)
                            skip_comments = True

                # 2. Genuine inline code-review comments (on diff lines)
                review_comments = []
                if not skip_comments:
                    try:
                        for rc in pr.get_review_comments():
                            review_comments.append({
                                "author": rc.user.login if rc.user else "unknown",
                                "user": rc.user.login if rc.user else "unknown",
                                "path": rc.path or "",
                                "line": rc.line or rc.original_line or 0,
                                "body": rc.body or "",
                                "created_at": rc.created_at.isoformat()
                            })
                    except Exception as e:
                        if "rate limit" in str(e).lower() or "403" in str(e) or "429" in str(e):
                            print("Warning: GitHub API rate limit hit while fetching PR review comments. Continuing with PR metadata only.", file=sys.stderr)
                            skip_comments = True

                prs_data.append({
                    "number": pr.number,
                    "title": pr.title,
                    "body": body_text,
                    "state": pr.state,
                    "author": pr.user.login if pr.user else "unknown",
                    "created_at": pr.created_at,
                    "merged_at": pr.merged_at,
                    "merge_commit_sha": pr.merge_commit_sha,
                    "comments": comments,
                    "review_comments": review_comments,
                    "linked_issue_numbers": linked_issues,
                })
                count += 1
        except Exception as e:
            print(f"Notice: Skipping remaining PR fetch due to GitHub API limit/error: {e}", file=sys.stderr)

        return prs_data

    def fetch_issues(
        self,
        state: str = "all",
        limit: int = 500,
        sort: str = "created",
        direction: str = "desc",
        since: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Fetches issues (direction='desc' by default) with comments, labels, and linked PRs."""
        if not self.repo:
            return []

        issues_data = []
        try:
            rate_obj = getattr(self.gh.get_rate_limit(), 'core', None) or getattr(self.gh.get_rate_limit(), 'rate', None)
            rem = getattr(rate_obj, 'remaining', 0) if rate_obj else 0

            if rem < 5:
                print(f"Notice: GitHub API rate limit low ({rem} remaining). Skipping detailed Issue comments.", file=sys.stderr)
                return []

            kwargs = {"state": state, "sort": sort, "direction": direction}
            if since and isinstance(since, datetime):
                kwargs["since"] = since
            issues = self.repo.get_issues(**kwargs)
            count = 0
            skip_comments = not bool(self.token)
            for issue in issues:

                if count >= limit:
                    break
                if since and issue.created_at:
                    issue_c = issue.created_at.replace(tzinfo=None) if hasattr(issue.created_at, 'tzinfo') and issue.created_at.tzinfo else issue.created_at
                    since_cmp = since.replace(tzinfo=None) if hasattr(since, 'tzinfo') and since.tzinfo else since
                    if issue_c < since_cmp:
                        break
                if issue.pull_request:  # Skip PRs returned by issue endpoint
                    continue

                body_text = issue.body or ""
                linked_prs = [int(n) for n in re.findall(r'(?:pr|pull request|see)\s+#(\d+)', body_text, re.IGNORECASE)]

                comments = []
                if not skip_comments:
                    try:
                        for c in issue.get_comments():
                            comments.append({
                                "author": c.user.login if c.user else "unknown",
                                "user": c.user.login if c.user else "unknown",
                                "body": c.body or "",
                                "created_at": c.created_at.isoformat()
                            })
                    except Exception as e:
                        if "rate limit" in str(e).lower() or "403" in str(e) or "429" in str(e):
                            print("Warning: GitHub API rate limit hit while fetching issue comments. Continuing with Issue metadata only.", file=sys.stderr)
                            skip_comments = True

                issues_data.append({
                    "number": issue.number,
                    "title": issue.title,
                    "body": body_text,
                    "state": issue.state,
                    "author": issue.user.login if issue.user else "unknown",
                    "created_at": issue.created_at,
                    "closed_at": issue.closed_at,
                    "labels": [l.name for l in issue.labels],
                    "comments": comments,
                    "linked_pr_numbers": linked_prs,
                })
                count += 1
        except Exception as e:
            print(f"Notice: Skipping remaining Issue fetch due to GitHub API limit/error: {e}", file=sys.stderr)

        return issues_data

