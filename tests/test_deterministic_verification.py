"""Regression and unit tests for the 4-Stage Hybrid Deterministic Verification Engine.

Covers:
- Stage 1: Pre-Draft Evidence Sufficiency & Causal Link Auto-Hydration (scoped by repo_id)
- Stage 2: SQLite & Chunk Relational Tuple Verification (Cross-Commit File/Symbol Attribution,
           Clause Splitting for Compound Sentences, Directional Revert Integrity, Chronology)
- Stage 3: Structural Code Identifier Grounding, English Hex-Word Safety, and Deterministic
           Fast-Pass (skipping redundant LLM Judge calls)
- Stage 4: Post-Synthesis Provenance & Citation Completeness Guardrail in `synthesize_node`
"""

from datetime import datetime
from unittest.mock import MagicMock

from sqlmodel import Session

from archaeologist.agent.nodes.synthesize import synthesize_node
from archaeologist.agent.nodes.verify import verify_node
from archaeologist.agent.verification_engine import (
    build_evidence_index,
    extract_cited_shas,
    hydrate_missing_causal_links,
    verify_claims_deterministically,
)
from archaeologist.storage.context import current_db_url_var
from archaeologist.storage.db import get_engine, init_db
from archaeologist.storage.models import Commit, PullRequest


def test_stage1_auto_hydration_of_missing_causal_links_scoped_by_repo(tmp_path):
    """Stage 1: Missing linked PRs and reverted commits referenced by top chunks are
    deterministically hydrated from SQLite and strictly scoped by repo_id.
    """
    db_file = tmp_path / "stage1_hydrate.db"
    db_url = f"sqlite:///{db_file}"
    token = current_db_url_var.set(db_url)
    try:
        init_db(db_url)
        engine = get_engine(db_url)
        with Session(engine) as session:
            # Target repo entities
            session.add(
                PullRequest(
                    repo_id="org/repo_a",
                    number=421,
                    title="Add exponential backoff to fetchUser",
                    body="Fixes upstream 503 timeouts by retrying 3 times.",
                    state="merged",
                    author="alice",
                    created_at=datetime(2025, 1, 10, 12, 0, 0),
                    linked_issue_numbers=[389],
                )
            )
            session.add(
                Commit(
                    repo_id="org/repo_a",
                    sha="2d24115abcdef1234567890abcdef1234567890",
                    author_name="bob",
                    author_email="bob@example.com",
                    authored_date=datetime(2025, 1, 5, 10, 0, 0),
                    message="Initial naive single-pass request logic",
                    files_changed=["src/user.py"],
                    symbols_modified=["fetchUser"],
                )
            )
            # Cross-tenant decoy PR with same number in repo_b
            session.add(
                PullRequest(
                    repo_id="org/repo_b",
                    number=421,
                    title="LEAKED REPO_B PR TITLE",
                    body="Should never be hydrated into repo_a",
                    state="merged",
                    author="mallory",
                    created_at=datetime(2025, 1, 10, 12, 0, 0),
                )
            )
            session.commit()

        initial_chunks = [
            {
                "id": "seed_commit_1",
                "source_type": "commit",
                "source_id": "8f31a2b999999999999999999999999999999999",
                "text": "Commit 8f31a2b: Handle transient 503 failures in fetchUser",
                "related_ids": ["pr#421"],
                "reverts_sha": "2d24115abcdef1234567890abcdef1234567890",
                "rrf_score": 0.050,
                "sub_question_idx": 0,
            }
        ]

        hydrated = hydrate_missing_causal_links(initial_chunks, repo_id="org/repo_a", max_hydrate=3)
        assert len(hydrated) == 3

        hydrated_texts = "\n".join(c["text"] for c in hydrated)
        assert "Add exponential backoff to fetchUser" in hydrated_texts
        assert "Initial naive single-pass request logic" in hydrated_texts
        assert "LEAKED REPO_B PR TITLE" not in hydrated_texts

        # Verify inherited score decay (0.050 * 0.90 = 0.045)
        pr_chunk = next(c for c in hydrated if c["source_id"] == "pr#421")
        assert abs(pr_chunk["rrf_score"] - 0.045) < 1e-5
    finally:
        current_db_url_var.reset(token)


def test_stage2_cross_commit_misattribution_and_compound_sentence_safety(monkeypatch):
    """Stage 2: Catches cross-commit file & symbol misattribution (Chimera hallucination)
    and short-circuits the LLM Judge (0 generate_json calls), while allowing valid
    compound sentences that distinguish between two commits.
    """
    import archaeologist.agent.nodes.verify as vm

    chunks = [
        {
            "id": "c_auth",
            "source_type": "commit",
            "source_id": "8f31a2b111111111111111111111111111111111",
            "text": "Commit 8f31a2b: Harden token validation in src/auth.py (validate_token)",
            "file_paths": ["src/auth.py"],
            "symbols_modified": ["validate_token"],
            "related_ids": ["pr#101"],
            "rrf_score": 0.04,
        },
        {
            "id": "c_sess",
            "source_type": "commit",
            "source_id": "4c99e10222222222222222222222222222222222",
            "text": "Commit 4c99e10: Rotate session cookies in src/session.py (refresh_cookie)",
            "file_paths": ["src/session.py"],
            "symbols_modified": ["refresh_cookie"],
            "related_ids": ["pr#202"],
            "rrf_score": 0.035,
        },
    ]

    judge_json_calls = 0

    class SpyGeminiClient:
        def __init__(self, *args, **kwargs):
            pass

        def generate_text(self, *args, **kwargs):
            # Misattributes src/session.py and `refresh_cookie` to commit 8f31a2b
            return "In PR #101, commit 8f31a2b modified src/session.py and updated `refresh_cookie`."

        def generate_json(self, *args, **kwargs):
            nonlocal judge_json_calls
            judge_json_calls += 1
            return {"verification_passed": True, "unverified_claims": []}

    monkeypatch.setattr(vm, "GeminiClientWrapper", SpyGeminiClient)
    monkeypatch.setattr(vm, "get_gemini_api_key", lambda: "test_key")

    out = verify_node(
        {
            "question": "Why did auth and session change?",
            "retrieved_chunks": chunks,
            "draft_answer": None,
            "verification_passed": False,
        }
    )

    # Must deterministically reject WITHOUT calling generate_json
    assert out["verification_passed"] is False
    assert judge_json_calls == 0
    claims_joined = " | ".join(out["unverified_claims"])
    assert "Cross-commit file misattribution" in claims_joined
    assert "Cross-commit symbol misattribution" in claims_joined

    # Now verify that a valid compound sentence with both commits passes cleanly (Pitfall 2 check)
    ev_text = "\n".join(c["text"] for c in chunks)
    ev_index = build_evidence_index(chunks, ev_text)
    valid_compound = (
        "Commit 8f31a2b modified src/auth.py and `validate_token` in PR #101, "
        "while commit 4c99e10 updated src/session.py and `refresh_cookie` in PR #202."
    )
    det_valid = verify_claims_deterministically(valid_compound, ev_index, question="Why did auth and session change?")
    assert det_valid.passed is True
    assert det_valid.unverified_claims == []
    assert det_valid.fast_pass is True


def test_stage2_directional_revert_and_chronology_inversion(tmp_path):
    """Stage 2: Detects inverted revert direction ('A reverted B' when B actually reverted A)
    and chronological sequence inversion ('B preceded A' when A is older than B).
    """
    db_file = tmp_path / "stage2_revert_chrono.db"
    db_url = f"sqlite:///{db_file}"
    token = current_db_url_var.set(db_url)
    try:
        init_db(db_url)
        engine = get_engine(db_url)
        with Session(engine) as session:
            session.add(
                Commit(
                    repo_id="org/repo",
                    sha="aaaa111000000000000000000000000000000000",
                    author_name="dev",
                    author_email="dev@example.com",
                    authored_date=datetime(2024, 1, 10, 9, 0, 0),
                    message="Introduce experimental async pool",
                    files_changed=["src/pool.py"],
                    superseded_by_sha="bbbb222000000000000000000000000000000000",
                )
            )
            session.add(
                Commit(
                    repo_id="org/repo",
                    sha="bbbb222000000000000000000000000000000000",
                    author_name="dev",
                    author_email="dev@example.com",
                    authored_date=datetime(2024, 2, 15, 14, 0, 0),
                    message='Revert "Introduce experimental async pool"',
                    files_changed=["src/pool.py"],
                    is_revert=True,
                    reverts_sha="aaaa111000000000000000000000000000000000",
                )
            )
            session.commit()

        chunks = [
            {
                "id": "c1",
                "source_type": "commit",
                "source_id": "aaaa111000000000000000000000000000000000",
                "text": "Commit aaaa111: Introduce experimental async pool",
            },
            {
                "id": "c2",
                "source_type": "commit",
                "source_id": "bbbb222000000000000000000000000000000000",
                "text": "Commit bbbb222: Revert experimental async pool",
                "reverts_sha": "aaaa111000000000000000000000000000000000",
            },
        ]
        ev_index = build_evidence_index(chunks, "Commit aaaa111 and Commit bbbb222", repo_id="org/repo")

        # Claiming aaaa111 reverted bbbb222 and bbbb222 preceded aaaa111 is backwards on both counts!
        bad_draft = "Commit aaaa111 reverted commit bbbb222. Commit bbbb222 preceded commit aaaa111."
        res = verify_claims_deterministically(bad_draft, ev_index, repo_id="org/repo")
        assert res.passed is False
        joined = " | ".join(res.unverified_claims)
        assert "Causal direction inversion" in joined
        assert "Chronological order inversion" in joined
    finally:
        current_db_url_var.reset(token)


def test_stage3_hex_english_words_and_ungrounded_code_identifiers():
    """Stage 3: English words made of [a-f] ('feedback', 'defaced', 'facade', 'scaffold')
    and pure numeric dates ('20240101') are never misclassified as SHAs, while invented
    snake_case/backticked code symbols are deterministically flagged.
    """
    prose = "In 20240101, user feedback on the defaced facade and scaffold was fixed in commit 8f31a2b."
    extracted = extract_cited_shas(prose)
    assert extracted == ["8f31a2b"]

    chunks = [
        {
            "id": "c1",
            "source_type": "commit",
            "source_id": "8f31a2b111111111111111111111111111111111",
            "text": "Commit 8f31a2b: Added retry_backoff_handler to src/client.py in PR #42",
            "related_ids": ["pr#42"],
        }
    ]
    ev_index = build_evidence_index(chunks, chunks[0]["text"])

    # Draft inventing `quantum_hyper_warp_authenticator` which does not exist in evidence
    invented_draft = (
        "In PR #42, commit 8f31a2b added `quantum_hyper_warp_authenticator` to src/client.py."
    )
    res_bad = verify_claims_deterministically(invented_draft, ev_index, question="Why did client change?")
    assert res_bad.passed is False
    assert any("quantum_hyper_warp_authenticator" in c for c in res_bad.unverified_claims)


def test_stage3_deterministic_fast_pass_skips_llm_judge(monkeypatch):
    """Stage 3: When a draft has verified citations, consistent relational tuples, and
    grounded code identifiers, `verify_node` fast-passes with ZERO calls to `generate_json`.
    """
    import archaeologist.agent.nodes.verify as vm

    judge_json_called = False

    class FastPassClient:
        def __init__(self, *args, **kwargs):
            pass

        def generate_text(self, *args, **kwargs):
            return (
                "Retry logic was added in commit 8f31a2b (PR #421) by modifying "
                "`retry_backoff_handler` in `src/client.py` to resolve Issue #389."
            )

        def generate_json(self, *args, **kwargs):
            nonlocal judge_json_called
            judge_json_called = True
            raise AssertionError("generate_json should NOT be called on a deterministic fast-pass!")

    monkeypatch.setattr(vm, "GeminiClientWrapper", FastPassClient)
    monkeypatch.setattr(vm, "get_gemini_api_key", lambda: "test_key")

    chunks = [
        {
            "id": "c1",
            "source_type": "commit",
            "source_id": "8f31a2b111111111111111111111111111111111",
            "text": (
                "Commit 8f31a2b: Add retry_backoff_handler in src/client.py "
                "(PR #421, fixes Issue #389)"
            ),
            "file_paths": ["src/client.py"],
            "symbols_modified": ["retry_backoff_handler"],
            "related_ids": ["pr#421", "issue#389"],
            "rrf_score": 0.05,
        }
    ]

    out = verify_node(
        {
            "question": "Why was retry_backoff_handler added to src/client.py?",
            "retrieved_chunks": chunks,
            "draft_answer": None,
            "verification_passed": False,
        }
    )

    assert out["verification_passed"] is True
    assert out["unverified_claims"] == []
    assert judge_json_called is False


def test_stage4_post_synthesis_guardrail_appends_missing_prs_and_flags_hallucinations(monkeypatch):
    """Stage 4: In `synthesize_node`, if the LLM omits a verified PR from top evidence or
    invents an unverified SHA/PR during synthesis, the post-synthesis guardrail deterministically
    appends the missing PR reference section and flags the unverified citation.
    """
    mock_client = MagicMock()
    # Synthesized text cites real commit 8f31a2b, invents PR #9999, and forgets linked PR #421
    mock_client.generate_text.return_value = (
        "The timeout retry mechanism was implemented in commit 8f31a2b and discussed in PR #9999."
    )

    monkeypatch.setattr("archaeologist.agent.nodes.synthesize.GeminiClientWrapper", lambda api_key: mock_client)
    monkeypatch.setattr("archaeologist.agent.nodes.synthesize.get_gemini_api_key", lambda: "fake_key")

    state = {
        "question": "Why was timeout retry implemented?",
        "repo_id": "test_repo",
        "retrieved_chunks": [
            {
                "id": "c1",
                "source_type": "commit",
                "source_id": "8f31a2b111111111111111111111111111111111",
                "text": "Commit 8f31a2b: Implement timeout retry mechanism",
                "related_ids": ["pr#421"],
                "rrf_score": 0.05,
            }
        ],
        "draft_answer": "Implemented in commit 8f31a2b (PR #421).",
        "verification_passed": True,
        "unverified_claims": [],
    }

    res = synthesize_node(state)
    final_resp = res["response"]

    # 1. Missing verified PR #421 from top chunk was deterministically appended
    assert "### Historical PRs & Linked Issues" in final_resp
    assert "PR #421" in final_resp

    # 2. Hallucinated PR #9999 introduced during synthesis was deterministically flagged
    assert "Provenance guardrail flagged unverified reference(s) (#9999)" in final_resp
