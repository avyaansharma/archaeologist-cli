import os
from typing import Dict, Any, List, Optional

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

REPOSITORIES: Dict[str, Dict[str, Any]] = {
    "requests": {
        "id": "requests",
        "name": "psf/requests",
        "title": "Python HTTP for Humans",
        "description": "The gold-standard Python HTTP library with 6,490 historical commits and complete connection pool lifecycle architecture.",
        "db_path": os.path.join(WORKSPACE_ROOT, "eval", "data", "requests.db"),
        "bm25_path": os.path.join(WORKSPACE_ROOT, "eval", "data", "requests_bm25.bin"),
        "total_commits": 6490,
        "total_chunks": 7163,
        "language": "Python",
        "stars": "52.4k",
        "starter_questions": [
            {
                "id": "req-1",
                "question": "How does Session.send() handle request preparation, redirect history tracking, adapter dispatching, and cookie persistence?",
                "category": "Architecture & Lifecycle",
                "difficulty": "Hard"
            },
            {
                "id": "req-2",
                "question": "How does HTTPAdapter manage urllib3 connection pools and per-host pool limits?",
                "category": "Networking & Transports",
                "difficulty": "Medium"
            },
            {
                "id": "req-3",
                "question": "Why does CaseInsensitiveDict preserve original key casing while supporting lower-case lookups?",
                "category": "Data Structures",
                "difficulty": "Medium"
            },
            {
                "id": "req-4",
                "question": "How does Response.raise_for_status() distinguish between 4xx client and 5xx server HTTP errors?",
                "category": "Error Handling",
                "difficulty": "Easy"
            }
        ]
    },
    "flask": {
        "id": "flask",
        "name": "pallets/flask",
        "title": "Lightweight WSGI Web Framework",
        "description": "The micro web framework for Python featuring ContextVar async context isolation, Blueprint route deferral, and CLI integration.",
        "db_path": os.path.join(WORKSPACE_ROOT, "eval", "data", "flask.db"),
        "bm25_path": os.path.join(WORKSPACE_ROOT, "eval", "data", "flask_bm25.bin"),
        "total_commits": 673,
        "total_chunks": 1390,
        "language": "Python",
        "stars": "67.8k",
        "starter_questions": [
            {
                "id": "flask-1",
                "question": "How does Flask handle request context and application context isolation using ContextVar and LocalProxy in src/flask/globals.py?",
                "category": "Context Isolation",
                "difficulty": "Hard"
            },
            {
                "id": "flask-2",
                "question": "Why does Blueprint.record() defer route and error-handler registration until the application initializes in src/flask/blueprints.py?",
                "category": "Routing Architecture",
                "difficulty": "Hard"
            },
            {
                "id": "flask-3",
                "question": "How does Flask integrate with Click via FlaskGroup in src/flask/cli.py to automatically discover application instances?",
                "category": "CLI Subsystem",
                "difficulty": "Medium"
            },
            {
                "id": "flask-4",
                "question": "How does SessionInterface implement secure signed cookies using itsdangerous in src/flask/sessions.py?",
                "category": "Session & Security",
                "difficulty": "Medium"
            },
            {
                "id": "flask-5",
                "question": "How does full_dispatch_request() manage error handling and teardown lifecycle in src/flask/app.py?",
                "category": "Request Pipeline",
                "difficulty": "Hard"
            }
        ]
    },
    "mss": {
        "id": "mss",
        "name": "BoboTiG/python-mss",
        "title": "Ultra-Fast Screen Shot Library",
        "description": "An ultra-fast, cross-platform pure-Python screenshot library with 1,053 historical commits, X11 shared memory, and GDI backends.",
        "db_path": os.path.join(WORKSPACE_ROOT, "eval", "data", "mss.db"),
        "bm25_path": os.path.join(WORKSPACE_ROOT, "eval", "data", "mss_bm25.bin"),
        "total_commits": 1053,
        "total_chunks": 2514,
        "language": "Python / Ctypes",
        "stars": "2.3k",
        "starter_questions": [
            {
                "id": "mss-1",
                "question": "A fix was written in commit 5e5f3ee for Windows region caching and reverted in commit 2d24115. Why was it reverted, and what happened to the test?",
                "category": "Causal Revert History",
                "difficulty": "Hard"
            },
            {
                "id": "mss-2",
                "question": "In PR #452 (commit 06dc845), why did the library move from a single global lock to per-object locking, and which backend kept a global lock?",
                "category": "Concurrency & Locking",
                "difficulty": "Medium"
            },
            {
                "id": "mss-3",
                "question": "In PR #467 (commit 0822b33) and PR #468 (commit 9637209), how was the KeyboardInterrupt bug during buffer copying resolved using memoryviews?",
                "category": "Memory & Signals",
                "difficulty": "Hard"
            },
            {
                "id": "mss-4",
                "question": "In PR #494 / Issue #486 (commit 8a7bbc2), why was the library redesigned with a single top-level MSS class, and what got deprecated?",
                "category": "Architecture Redesign",
                "difficulty": "Medium"
            }
        ]
    }
}

def get_repo_config(repo_id: str) -> Optional[Dict[str, Any]]:
    """Returns repository configuration by ID (case-insensitive)."""
    return REPOSITORIES.get(repo_id.lower().strip())

def list_repo_configs() -> List[Dict[str, Any]]:
    """Returns all configured repository summaries."""
    configs = []
    for k, v in REPOSITORIES.items():
        item = dict(v)
        if "id" not in item:
            item["id"] = item.get("repo_id", k)
        configs.append(item)
    return configs
