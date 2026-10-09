import pytest
from archaeologist.ingestion.link_resolver import extract_refs, update_cross_links

def test_extract_refs():
    text = "Fixes #123, closes #456. Also checked https://github.com/owner/repo/pull/789. Ref abc1234."
    refs = extract_refs(text)
    
    assert refs["closes"] == [123, 456]
    assert refs["mentions"] == [789]
    assert "abc1234" in refs["shas"]

def test_update_cross_links():
    prs = [
        {
            "number": 10,
            "title": "Fix retry login bug",
            "body": "This fixes #20 and mentions #25",
            "author": "john",
            "review_comments": []
        }
    ]
    
    issues = [
        {
            "number": 20,
            "title": "Retry login fails on rate limits",
            "body": "Need to add backoff",
            "comments": []
        },
        {
            "number": 25,
            "title": "Docs update for login",
            "body": "",
            "comments": []
        }
    ]
    
    updated_prs, updated_issues = update_cross_links(prs, issues)
    
    pr_10 = updated_prs[0]
    assert 20 in pr_10["linked_issue_numbers"]
    assert 25 in pr_10["linked_issue_numbers"]
    
    issue_20 = next(i for i in updated_issues if i["number"] == 20)
    assert 10 in issue_20["linked_pr_numbers"]
    
    issue_25 = next(i for i in updated_issues if i["number"] == 25)
    assert 10 in issue_25["linked_pr_numbers"]
