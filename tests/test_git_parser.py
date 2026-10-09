from archaeologist.ingestion.git_parser import _parse_commit_record

def test_parse_commit_record():
    FIELD_DELIM = "\x1f"
    
    header = FIELD_DELIM.join([
        "a1b2c3d4e5f6g7h8i9j0",
        "Jane Doe",
        "jane@example.com",
        "2026-08-14T12:34:56+00:00",
        "Fix bug in auth module\n\nResolves #101 and references #102."
    ])
    
    numstat = "10\t5\tsrc/auth.py\n2\t0\tsrc/utils.py"
    
    record = f"{header}{FIELD_DELIM}NUMSTAT\n{numstat}"
    
    parsed = _parse_commit_record(record)
    
    assert parsed["sha"] == "a1b2c3d4e5f6g7h8i9j0"
    assert parsed["author_name"] == "Jane Doe"
    assert parsed["author_email"] == "jane@example.com"
    assert parsed["message"] == "Fix bug in auth module\n\nResolves #101 and references #102."
    assert parsed["files_changed"] == ["src/auth.py", "src/utils.py"]
    assert parsed["insertions"] == 12
    assert parsed["deletions"] == 5
