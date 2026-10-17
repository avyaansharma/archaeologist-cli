"""Deterministic multi-stage verification engine for Codebase Archaeologist.

Implements a 4-stage hybrid verification pipeline that validates historical claims
against in-memory evidence chunks and SQLite relational tables (`Commit`, `PullRequest`,
`Issue`, `SymbolIndex`, `CausalEdge`) before falling back to an LLM judge:

- Stage 1: Pre-Draft Evidence Sufficiency & Missing Causal Link Auto-Hydration
- Stage 2: SQLite & Evidence Relational Tuple Verifier (Provenance, Cross-Commit
           File/Symbol Attribution, Directional Revert Integrity, Chronology)
- Stage 3: Structural Code Identifier Grounding & Deterministic Fast-Pass Gate
- Stage 4: Post-Synthesis Provenance & Citation Completeness Guardrail
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlmodel import select

from archaeologist.storage.db import get_session_context
from archaeologist.storage.models import (
    Commit,
    Issue,
    PullRequest,
    _repo_filter,
    is_valid_commit_sha,
    is_valid_pr_or_issue,
)

# Common English words composed solely of hex characters [a-f] (7+ chars) that must
# never be misclassified as git commit SHAs unless preceded by "commit ".
_HEX_ENGLISH_WORDS = frozenset({
    "acceded",
    "cabbed",
    "dabbed",
    "deafened",
    "debrief",
    "decaf",
    "deface",
    "defaced",
    "efface",
    "effaced",
    "facade",
    "facaded",
    "feedback",
    "scaffold",
})

_MODIFICATION_VERBS_RE = re.compile(
    r"\b(?:modified|modifies|changed|changes|updated|updates|added|adds|"
    r"introduced|introduces|refactored|refactors|removed|removes|deleted|deletes|"
    r"implemented|implements|fixed|fixes|touched|touches|patched|patches)\b",
    re.IGNORECASE,
)

_SOURCE_FILE_RE = re.compile(
    r"\b([A-Za-z0-9_./\\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|c|cpp|h|hpp|rb|php|cs|swift|kt|scala|sql|sh|toml|yaml|yml|json))\b"
)

_CLAUSE_SPLIT_RE = re.compile(
    r"(?:(?:\.(?=\s|$)|[;\n])+|\s*,\s*(?:while|whereas|and\s+commit|but\s+commit)\s+)",
    re.IGNORECASE,
)


@dataclass
class EvidenceIndex:
    """Unified ground-truth index combining retrieved chunks and scoped SQLite records."""

    known_shas: set[str] = field(default_factory=set)
    known_prs_issues: set[str] = field(default_factory=set)
    commit_footprints: dict[str, dict[str, Any]] = field(default_factory=dict)
    pr_footprints: dict[str, dict[str, Any]] = field(default_factory=dict)
    known_files: set[str] = field(default_factory=set)
    known_symbols: set[str] = field(default_factory=set)
    evidence_text_lower: str = ""

    def get_commit(self, sha_str: str) -> dict[str, Any] | None:
        """Look up a commit footprint by full SHA or 7+ char prefix."""
        sha_l = sha_str.lower().strip()
        if len(sha_l) < 7:
            return None
        prefix = sha_l[:7]
        if prefix in self.commit_footprints:
            return self.commit_footprints[prefix]
        for k, fp in self.commit_footprints.items():
            full_s = fp.get("full_sha", k)
            if full_s.startswith(sha_l) or sha_l.startswith(k):
                return fp
        return None

    def is_known_sha(self, sha_str: str) -> bool:
        sha_l = sha_str.lower().strip()
        if len(sha_l) < 7:
            return False
        if sha_l in self.known_shas or sha_l[:7] in self.known_shas:
            return True
        if any(
            k.startswith(sha_l[:7]) or sha_l.startswith(k[:7])
            for k in self.known_shas
            if len(k) >= 7
        ):
            return True
        return re.search(rf"\b{re.escape(sha_l)}\b", self.evidence_text_lower) is not None

    def is_known_pr_or_issue(self, num_str: str) -> bool:
        clean = str(num_str).strip().lstrip("#").lower()
        if clean in self.known_prs_issues:
            return True
        return (
            re.search(
                rf"\b(?:pr|pull\s+request|issue|#)\s*#?{re.escape(clean)}\b",
                self.evidence_text_lower,
                re.IGNORECASE,
            )
            is not None
        )


@dataclass
class DeterministicVerificationResult:
    """Structured outcome of Stages 2 and 3 deterministic verification."""

    passed: bool
    unverified_claims: list[str] = field(default_factory=list)
    verified_citation_count: int = 0
    grounding_score: float = 1.0
    fast_pass: bool = False


def extract_cited_shas(text: str, known_shas: set[str] | None = None) -> list[str]:
    """Extract commit SHA references from prose while avoiding dates, numbers, and English hex words."""
    if not text:
        return []
    known = known_shas or set()
    explicit_commit_shas = {
        m.lower()
        for m in re.findall(r"\bcommit\s+`?([0-9a-fA-F]{7,40})`?\b", text, re.IGNORECASE)
        if not m.isdigit()
    }

    results: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"(?<![0-9a-fA-FxX])([0-9a-fA-F]{7,40})\b", text):
        token = match.group(1)
        tok_l = token.lower()
        if tok_l in seen or token.isdigit():
            continue
        # Exclude hex literals like 0x1234567
        start_idx = match.start(1)
        if start_idx >= 2 and text[start_idx - 2 : start_idx].lower() == "0x":
            continue
        if tok_l in _HEX_ENGLISH_WORDS and tok_l not in explicit_commit_shas:
            continue

        has_digit = any(c.isdigit() for c in tok_l)
        has_hex_alpha = any(c in "abcdef" for c in tok_l)
        is_known = any(
            k.startswith(tok_l[:7]) or tok_l.startswith(k[:7])
            for k in known
            if len(k) >= 7
        )

        if len(tok_l) == 40 or (has_digit and has_hex_alpha) or is_known or tok_l in explicit_commit_shas:
            results.append(token)
            seen.add(tok_l)
    return results


def extract_cited_prs_issues(text: str) -> list[str]:
    """Extract cited PR or Issue numbers (e.g. 'PR #123', 'Issue #45', '#78')."""
    if not text:
        return []
    nums = re.findall(r"(?:PR|Pull\s+Request|Issue|#)\s*#?(\d+)", text, re.IGNORECASE)
    seen: set[str] = set()
    ordered: list[str] = []
    for n in nums:
        if n not in seen:
            ordered.append(n)
            seen.add(n)
    return ordered


def _normalize_file_token(fp: str) -> str:
    return fp.strip().strip("`'\".,;:()[]").replace("\\", "/").lower()


# ---------------------------------------------------------------------------
# STAGE 1: Pre-Draft Evidence Sufficiency & Causal Link Auto-Hydration
# ---------------------------------------------------------------------------

def hydrate_missing_causal_links(
    retrieved_chunks: list[dict[str, Any]],
    repo_id: str | None = None,
    max_hydrate: int = 3,
) -> list[dict[str, Any]]:
    """Stage 1: Deterministically hydrate missing linked PRs/Issues/Reverts referenced
    by the top retrieved chunks directly from SQLite before drafting.
    """
    if not retrieved_chunks or max_hydrate <= 0:
        return retrieved_chunks

    valid_chunks = [c for c in retrieved_chunks if isinstance(c, dict)]
    if not valid_chunks:
        return retrieved_chunks

    existing_source_ids = {str(c.get("source_id", "")).lower() for c in valid_chunks if c.get("source_id")}
    existing_chunk_ids = {str(c.get("id", "")) for c in valid_chunks if c.get("id")}

    top_chunks = sorted(
        valid_chunks,
        key=lambda c: float(c.get("rrf_score") or c.get("score") or 0.0),
        reverse=True,
    )[:5]

    missing_pr_nums: list[tuple[int, float, int | None]] = []
    missing_shas: list[tuple[str, float, int | None]] = []

    for c in top_chunks:
        parent_score = float(c.get("rrf_score") or c.get("score") or 0.02)
        sq_idx = c.get("sub_question_idx")
        for rel in c.get("related_ids") or []:
            rel_s = str(rel).lower()
            m_num = re.search(r"(\d+)", rel_s)
            if m_num:
                num = int(m_num.group(1))
                if (
                    f"pr#{num}" not in existing_source_ids
                    and f"issue#{num}" not in existing_source_ids
                    and str(num) not in existing_source_ids
                ):
                    missing_pr_nums.append((num, parent_score, sq_idx))
        rev_sha = c.get("reverts_sha")
        if isinstance(rev_sha, str) and len(rev_sha) >= 7:
            rev_l = rev_sha.lower()
            if not any(s.startswith(rev_l[:7]) or rev_l.startswith(s[:7]) for s in existing_source_ids if len(s) >= 7):
                missing_shas.append((rev_l, parent_score, sq_idx))

    if not missing_pr_nums and not missing_shas:
        return retrieved_chunks

    hydrated: list[dict[str, Any]] = []
    try:
        with get_session_context() as session:
            # 1. Hydrate missing PRs / Issues
            for num, p_score, sq_idx in missing_pr_nums:
                if len(hydrated) >= max_hydrate:
                    break
                pr_stmt = select(PullRequest).where(PullRequest.number == num)
                if repo_id:
                    pr_stmt = pr_stmt.where(_repo_filter(PullRequest.repo_id, repo_id))
                pr = session.exec(pr_stmt).first()
                if pr:
                    cid = f"hydrated_pr_{pr.number}"
                    if cid not in existing_chunk_ids:
                        body_preview = (pr.body or "")[:600]
                        hydrated.append({
                            "id": cid,
                            "source_type": "pr",
                            "source_id": f"pr#{pr.number}",
                            "text": f"PR #{pr.number}: {pr.title} (by {pr.author})\n{body_preview}",
                            "related_ids": [f"issue#{i}" for i in (pr.linked_issue_numbers or [])],
                            "rrf_score": max(0.012, round(p_score * 0.90, 6)),
                            "score": max(0.012, round(p_score * 0.90, 6)),
                            "sub_question_idx": sq_idx,
                        })
                        existing_chunk_ids.add(cid)
                        existing_source_ids.add(f"pr#{pr.number}")
                        continue

                iss_stmt = select(Issue).where(Issue.number == num)
                if repo_id:
                    iss_stmt = iss_stmt.where(_repo_filter(Issue.repo_id, repo_id))
                iss = session.exec(iss_stmt).first()
                if iss:
                    cid = f"hydrated_issue_{iss.number}"
                    if cid not in existing_chunk_ids:
                        body_preview = (iss.body or "")[:600]
                        hydrated.append({
                            "id": cid,
                            "source_type": "issue",
                            "source_id": f"issue#{iss.number}",
                            "text": f"Issue #{iss.number}: {iss.title} (by {iss.author})\n{body_preview}",
                            "related_ids": [f"pr#{p}" for p in (iss.linked_pr_numbers or [])],
                            "rrf_score": max(0.012, round(p_score * 0.90, 6)),
                            "score": max(0.012, round(p_score * 0.90, 6)),
                            "sub_question_idx": sq_idx,
                        })
                        existing_chunk_ids.add(cid)
                        existing_source_ids.add(f"issue#{iss.number}")

            # 2. Hydrate missing reverted commits
            for sha_prefix, p_score, sq_idx in missing_shas:
                if len(hydrated) >= max_hydrate:
                    break
                from archaeologist.utils.security import escape_like
                esc = escape_like(sha_prefix[:12])
                c_stmt = select(Commit).where(Commit.sha.like(f"{esc}%", escape="\\"))
                if repo_id:
                    c_stmt = c_stmt.where(_repo_filter(Commit.repo_id, repo_id))
                commit_obj = session.exec(c_stmt).first()
                if commit_obj:
                    cid = f"hydrated_commit_{commit_obj.sha[:7]}"
                    if cid not in existing_chunk_ids:
                        hydrated.append({
                            "id": cid,
                            "source_type": "commit",
                            "source_id": commit_obj.sha,
                            "text": (
                                f"Commit {commit_obj.sha[:7]} by {commit_obj.author_name}: "
                                f"{commit_obj.message}\nFiles: {', '.join(commit_obj.files_changed or [])}"
                            ),
                            "file_paths": list(commit_obj.files_changed or []),
                            "symbols_modified": list(commit_obj.symbols_modified or []),
                            "related_ids": [],
                            "rrf_score": max(0.012, round(p_score * 0.90, 6)),
                            "score": max(0.012, round(p_score * 0.90, 6)),
                            "sub_question_idx": sq_idx,
                        })
                        existing_chunk_ids.add(cid)
                        existing_source_ids.add(commit_obj.sha.lower())
    except Exception:
        # Never fail verification if SQLite is uninitialized in synthetic unit tests
        return retrieved_chunks

    if not hydrated:
        return retrieved_chunks
    return list(retrieved_chunks) + hydrated


# ---------------------------------------------------------------------------
# STAGE 2 & 3: EvidenceIndex Builder & Deterministic Claim Verifier
# ---------------------------------------------------------------------------

def _extract_symbol_parts(sym: str) -> set[str]:
    """Extract normalized symbol aliases from qualified AST symbol IDs
    (supports both 'path.py::Class::method' and 'path.py:Class.method').
    """
    s_low = str(sym).strip().strip("`'\"()").lower()
    if not s_low:
        return set()
    parts = {s_low}
    after_colon = s_low.split("::")[-1].split(":")[-1]
    if after_colon:
        parts.add(after_colon)
        for seg in re.split(r"[.:]+", after_colon):
            if seg and len(seg) >= 2:
                parts.add(seg)
    return parts


def build_evidence_index(
    sorted_chunks: list[dict[str, Any]],
    evidence_text: str,
    repo_id: str | None = None,
    candidate_text: str = "",
) -> EvidenceIndex:
    """Build a unified EvidenceIndex from retrieved chunks and scoped SQLite tables."""
    idx = EvidenceIndex(evidence_text_lower=(evidence_text or "").lower())

    candidate_shas: set[str] = set()
    candidate_pr_nums: set[int] = set()

    # 1. Populate from in-memory retrieved chunks
    for c in sorted_chunks:
        if not isinstance(c, dict):
            continue
        sid = str(c.get("source_id", "") or "").strip()
        stype = str(c.get("source_type", "commit") or "commit").lower()
        c_text = str(c.get("text", "") or "")
        c_files = {_normalize_file_token(f) for f in (c.get("file_paths") or []) if f}
        c_symbols = {str(s).strip() for s in (c.get("symbols_modified") or []) if s}

        # Also extract file paths mentioned inside chunk text
        for fm in _SOURCE_FILE_RE.findall(c_text):
            c_files.add(_normalize_file_token(fm))

        idx.known_files.update(c_files)
        for f_path in list(c_files):
            idx.known_files.add(os.path.basename(f_path))

        for sym in c_symbols:
            idx.known_symbols.update(_extract_symbol_parts(sym))

        if sid:
            sid_l = sid.lower()
            if stype == "commit" or (len(sid_l) >= 7 and not sid_l.startswith(("pr#", "issue#"))):
                idx.known_shas.add(sid_l)
                idx.known_shas.add(sid_l[:7])
                candidate_shas.add(sid_l[:12])
                prefix = sid_l[:7]
                fp = idx.commit_footprints.setdefault(
                    prefix,
                    {
                        "full_sha": sid_l,
                        "files": set(),
                        "symbols": set(),
                        "prs": set(),
                        "reverts_sha": None,
                        "superseded_by_sha": None,
                        "authored_date": None,
                        "text": "",
                    },
                )
                fp["files"].update(c_files)
                for sym in c_symbols:
                    fp["symbols"].update(_extract_symbol_parts(sym))
                fp["text"] = (fp["text"] + "\n" + c_text).strip()
                if c.get("reverts_sha"):
                    fp["reverts_sha"] = str(c["reverts_sha"]).lower()

            m_pr_sid = re.search(r"(?:pr|issue)#?(\d+)", sid_l)
            if m_pr_sid:
                num_s = m_pr_sid.group(1)
                idx.known_prs_issues.add(num_s)
                idx.known_prs_issues.add(sid_l)
                candidate_pr_nums.add(int(num_s))

        for r in c.get("related_ids") or []:
            r_str = str(r).lower().strip()
            idx.known_prs_issues.add(r_str)
            m_rel = re.search(r"(\d+)", r_str)
            if m_rel:
                idx.known_prs_issues.add(m_rel.group(1))
                candidate_pr_nums.add(int(m_rel.group(1)))
                if sid and len(sid) >= 7 and sid.lower()[:7] in idx.commit_footprints:
                    idx.commit_footprints[sid.lower()[:7]]["prs"].add(m_rel.group(1))

    # Extract any additional SHAs / PRs mentioned inside evidence_text or candidate_text
    combined_scan = f"{evidence_text}\n{candidate_text}"
    for sha_cand in extract_cited_shas(combined_scan, idx.known_shas):
        candidate_shas.add(sha_cand.lower()[:12])
    for pr_cand in extract_cited_prs_issues(combined_scan):
        try:
            candidate_pr_nums.add(int(pr_cand))
        except ValueError:
            pass

    # 2. Enrich with authoritative relational rows from SQLite (when available)
    try:
        from archaeologist.utils.security import escape_like

        with get_session_context() as session:
            for sha_prefix in list(candidate_shas)[:25]:
                if len(sha_prefix) < 7:
                    continue
                esc = escape_like(sha_prefix[:12])
                stmt = select(Commit).where(Commit.sha.like(f"{esc}%", escape="\\"))
                if repo_id:
                    stmt = stmt.where(_repo_filter(Commit.repo_id, repo_id))
                commit_row = session.exec(stmt).first()
                if commit_row:
                    c_sha = commit_row.sha.lower()
                    idx.known_shas.add(c_sha)
                    idx.known_shas.add(c_sha[:7])
                    prefix = c_sha[:7]
                    fp = idx.commit_footprints.setdefault(
                        prefix,
                        {
                            "full_sha": c_sha,
                            "files": set(),
                            "symbols": set(),
                            "prs": set(),
                            "reverts_sha": None,
                            "superseded_by_sha": None,
                            "authored_date": None,
                            "text": "",
                        },
                    )
                    fp["full_sha"] = c_sha
                    for f_changed in commit_row.files_changed or []:
                        norm_f = _normalize_file_token(f_changed)
                        fp["files"].add(norm_f)
                        idx.known_files.add(norm_f)
                        idx.known_files.add(os.path.basename(norm_f))
                    for sym_mod in commit_row.symbols_modified or []:
                        sym_parts = _extract_symbol_parts(sym_mod)
                        fp["symbols"].update(sym_parts)
                        idx.known_symbols.update(sym_parts)
                    if commit_row.reverts_sha:
                        rev_s = commit_row.reverts_sha.lower()
                        fp["reverts_sha"] = rev_s
                        idx.known_shas.add(rev_s)
                        idx.known_shas.add(rev_s[:7])
                    if commit_row.superseded_by_sha:
                        sup_s = commit_row.superseded_by_sha.lower()
                        fp["superseded_by_sha"] = sup_s
                        idx.known_shas.add(sup_s)
                        idx.known_shas.add(sup_s[:7])
                    if isinstance(commit_row.authored_date, datetime):
                        fp["authored_date"] = commit_row.authored_date
                    if commit_row.message:
                        fp["text"] = (fp["text"] + "\n" + commit_row.message).strip()

            for pr_num in list(candidate_pr_nums)[:25]:
                if is_valid_pr_or_issue(session, pr_num, repo_id=repo_id):
                    idx.known_prs_issues.add(str(pr_num))
    except Exception:
        pass

    return idx


def _file_matches_footprint(file_token: str, fp: dict[str, Any]) -> bool:
    """Return True if file_token matches any file in the commit's footprint or chunk text."""
    norm = _normalize_file_token(file_token)
    base = os.path.basename(norm)
    fp_files: set[str] = fp.get("files") or set()
    for known_f in fp_files:
        if known_f == norm or known_f.endswith("/" + norm) or norm.endswith("/" + known_f):
            return True
        if os.path.basename(known_f) == base:
            return True
    fp_text = (fp.get("text") or "").lower()
    if norm in fp_text or base in fp_text:
        return True
    return False


def _symbol_matches_footprint(sym_token: str, fp: dict[str, Any]) -> bool:
    """Return True if sym_token matches any AST symbol or text in the commit's footprint."""
    cand_parts = _extract_symbol_parts(sym_token)
    fp_syms: set[str] = fp.get("symbols") or set()
    if cand_parts & fp_syms:
        return True
    sym_l = sym_token.strip("`'\"()").lower()
    short_sym = sym_l.split("::")[-1].split(":")[-1].split(".")[-1]
    fp_text = (fp.get("text") or "").lower()
    if sym_l in fp_text or (short_sym and short_sym in fp_text):
        return True
    return False


def verify_claims_deterministically(
    draft_text: str,
    evidence_index: EvidenceIndex,
    question: str = "",
    repo_id: str | None = None,
) -> DeterministicVerificationResult:
    """Execute Stage 2 (Relational Tuple & Provenance Verification) and
    Stage 3 (Structural Code Identifier Grounding & Fast-Pass Calculation).
    """
    if not draft_text or not draft_text.strip():
        return DeterministicVerificationResult(
            passed=False,
            unverified_claims=["Draft answer is empty."],
            verified_citation_count=0,
            grounding_score=0.0,
            fast_pass=False,
        )

    unverified: list[str] = []
    verified_citations = 0

    # ------------------------------------------------------------------
    # STAGE 2.1: Provenance Existence Check (PRs, Issues, Commit SHAs)
    # ------------------------------------------------------------------
    cited_prs = extract_cited_prs_issues(draft_text)
    for pr_num in cited_prs:
        if evidence_index.is_known_pr_or_issue(pr_num):
            verified_citations += 1
        else:
            unverified.append(
                f"Cited reference #{pr_num} not found in retrieved repository evidence."
            )

    cited_shas = extract_cited_shas(draft_text, evidence_index.known_shas)
    for sha in cited_shas:
        if evidence_index.is_known_sha(sha):
            verified_citations += 1
        else:
            # Double-check SQLite directly in case the SHA exists in the repo DB
            valid_in_db = False
            try:
                with get_session_context() as session:
                    valid_in_db = is_valid_commit_sha(session, sha.lower(), repo_id=repo_id)
            except Exception:
                valid_in_db = False
            if valid_in_db:
                verified_citations += 1
                evidence_index.known_shas.add(sha.lower())
                evidence_index.known_shas.add(sha.lower()[:7])
            else:
                unverified.append(
                    f"Cited commit SHA {sha} not found in retrieved repository evidence."
                )

    # ------------------------------------------------------------------
    # STAGE 2.2: Directional Revert Integrity Check
    # ------------------------------------------------------------------
    # Active voice: "Commit A reverted Commit B"
    active_reverts = re.findall(
        r"(?:commit\s+)?`?([0-9a-fA-F]{7,40})`?\s+(?:reverted|reverts|undid)\s+(?:the\s+changes\s+in\s+)?(?:commit\s+)?`?([0-9a-fA-F]{7,40})`?",
        draft_text,
        re.IGNORECASE,
    )
    # Passive voice: "Commit B was reverted by Commit A" / "Commit B was superseded by Commit A"
    passive_reverts = [
        (reverter, target)
        for target, reverter in re.findall(
            r"(?:commit\s+)?`?([0-9a-fA-F]{7,40})`?\s+was\s+(?:reverted|superseded|undone)\s+by\s+(?:commit\s+)?`?([0-9a-fA-F]{7,40})`?",
            draft_text,
            re.IGNORECASE,
        )
    ]

    for reverter_sha, target_sha in active_reverts + passive_reverts:
        rev_fp = evidence_index.get_commit(reverter_sha)
        tgt_fp = evidence_index.get_commit(target_sha)
        rev_prefix = reverter_sha.lower()[:7]
        tgt_prefix = target_sha.lower()[:7]

        # Check if the actual relationship in SQLite/evidence is the exact opposite
        if tgt_fp and tgt_fp.get("reverts_sha") and str(tgt_fp["reverts_sha"]).startswith(rev_prefix):
            unverified.append(
                f"Causal direction inversion: Commit {tgt_prefix} reverted {rev_prefix}, "
                f"not {rev_prefix} reverting {tgt_prefix}."
            )
        elif rev_fp and rev_fp.get("superseded_by_sha") and str(rev_fp["superseded_by_sha"]).startswith(tgt_prefix):
            unverified.append(
                f"Causal direction inversion: Commit {rev_prefix} was superseded/reverted by {tgt_prefix}, "
                f"not {rev_prefix} reverting {tgt_prefix}."
            )
        elif rev_fp and rev_fp.get("reverts_sha"):
            actual_tgt = str(rev_fp["reverts_sha"]).lower()
            if not (actual_tgt.startswith(tgt_prefix) or tgt_prefix.startswith(actual_tgt[:7])):
                unverified.append(
                    f"Relational revert mismatch: Commit {rev_prefix} reverted {actual_tgt[:7]}, "
                    f"not {tgt_prefix}."
                )

    # ------------------------------------------------------------------
    # STAGE 2.3: Clause-Scoped Cross-Commit File & Symbol Attribution
    # ------------------------------------------------------------------
    clauses = [cl.strip() for cl in _CLAUSE_SPLIT_RE.split(draft_text) if cl and cl.strip()]
    for clause in clauses:
        clause_shas = extract_cited_shas(clause, evidence_index.known_shas)
        known_clause_shas = [s for s in clause_shas if evidence_index.get_commit(s) is not None]

        # Single-subject clause attribution check
        if len(known_clause_shas) == 1 and _MODIFICATION_VERBS_RE.search(clause):
            subj_sha = known_clause_shas[0]
            subj_prefix = subj_sha.lower()[:7]
            subj_fp = evidence_index.get_commit(subj_sha)
            assert subj_fp is not None

            # 1) File attribution check
            clause_files = _SOURCE_FILE_RE.findall(clause)
            if clause_files and subj_fp.get("files"):
                for cf in clause_files:
                    if not _file_matches_footprint(cf, subj_fp):
                        # Check if this file actually belongs to a DIFFERENT commit in the evidence
                        other_owners = [
                            other_k
                            for other_k, other_fp in evidence_index.commit_footprints.items()
                            if other_k != subj_prefix and _file_matches_footprint(cf, other_fp)
                        ]
                        if other_owners:
                            subj_files_sample = sorted(subj_fp["files"])[:3]
                            unverified.append(
                                f"Cross-commit file misattribution: Commit {subj_prefix} modified "
                                f"{subj_files_sample}, not '{cf}' (which was modified in Commit {other_owners[0]})."
                            )

            # 2) Backticked symbol attribution check
            backticked_syms = [
                s
                for s in re.findall(r"`([A-Za-z_][A-Za-z0-9_.:]{2,})`", clause)
                if not _SOURCE_FILE_RE.match(s) and s.lower() != subj_sha.lower()
            ]
            if backticked_syms and subj_fp.get("symbols"):
                for bsym in backticked_syms:
                    if not _symbol_matches_footprint(bsym, subj_fp):
                        other_sym_owners = [
                            other_k
                            for other_k, other_fp in evidence_index.commit_footprints.items()
                            if other_k != subj_prefix and _symbol_matches_footprint(bsym, other_fp)
                        ]
                        if other_sym_owners:
                            unverified.append(
                                f"Cross-commit symbol misattribution: Commit {subj_prefix} did not modify "
                                f"symbol '{bsym}' (modified in Commit {other_sym_owners[0]})."
                            )

        # --------------------------------------------------------------
        # STAGE 2.4: Chronological Sequence Integrity
        # --------------------------------------------------------------
        chrono_match = re.search(
            r"(?:commit\s+)?`?([0-9a-fA-F]{7,40})`?\s+(?:preceded|came\s+before|occurred\s+before|was\s+committed\s+before)\s+(?:commit\s+)?`?([0-9a-fA-F]{7,40})`?",
            clause,
            re.IGNORECASE,
        )
        if chrono_match:
            earlier_s, later_s = chrono_match.group(1), chrono_match.group(2)
            fp_early = evidence_index.get_commit(earlier_s)
            fp_late = evidence_index.get_commit(later_s)
            if (
                fp_early
                and fp_late
                and isinstance(fp_early.get("authored_date"), datetime)
                and isinstance(fp_late.get("authored_date"), datetime)
            ):
                if fp_early["authored_date"] > fp_late["authored_date"]:
                    unverified.append(
                        f"Chronological order inversion: Commit {earlier_s[:7]} "
                        f"({fp_early['authored_date'].date()}) occurred after Commit {later_s[:7]} "
                        f"({fp_late['authored_date'].date()}), not before."
                    )

    # ------------------------------------------------------------------
    # STAGE 3: Structural Code Identifier Grounding & Fast-Pass Gate
    # ------------------------------------------------------------------
    question_lower = (question or "").lower()
    ev_lower = evidence_index.evidence_text_lower

    # Extract explicit structural code identifiers (snake_case & backticked code identifiers)
    backticked_identifiers = [
        m.strip("()")
        for m in re.findall(r"`([A-Za-z_][A-Za-z0-9_.()]{2,})`", draft_text)
    ]
    snake_identifiers = re.findall(r"\b([a-z][a-z0-9]*_[a-z0-9_]{2,})\b", draft_text)

    candidate_code_tokens: list[str] = []
    seen_toks: set[str] = set()
    for raw_tok in backticked_identifiers + snake_identifiers:
        clean_tok = raw_tok.strip("`'\"().,;:")
        tok_l = clean_tok.lower()
        if not clean_tok or tok_l in seen_toks:
            continue
        # Skip commit SHAs or PR refs
        if any(tok_l == s.lower() for s in cited_shas) or re.match(r"^(?:pr|issue)#?\d+$", tok_l):
            continue
        seen_toks.add(tok_l)
        candidate_code_tokens.append(clean_tok)

    grounded_count = 0
    ungrounded_tokens: list[str] = []

    for tok in candidate_code_tokens:
        tok_l = tok.lower()
        short_part = tok_l.split("::")[-1].split(".")[-1]
        sub_parts = [p for p in re.split(r"[_.:/]", tok_l) if len(p) >= 3]

        is_grounded = (
            tok_l in ev_lower
            or short_part in ev_lower
            or tok_l in question_lower
            or short_part in question_lower
            or tok_l in evidence_index.known_symbols
            or short_part in evidence_index.known_symbols
            or tok_l in evidence_index.known_files
            or (len(sub_parts) >= 2 and all(sp in ev_lower or sp in question_lower for sp in sub_parts))
        )
        if is_grounded:
            grounded_count += 1
        else:
            ungrounded_tokens.append(tok)

    # Flag invented backticked/snake_case symbols only when evidence_text is non-empty
    if ev_lower and ungrounded_tokens:
        # Only fail if more than 1 structural code token is completely ungrounded, or if a
        # single multi-segment snake_case/dotted symbol is completely absent from evidence & question
        severe_ungrounded = [
            t for t in ungrounded_tokens if ("_" in t or "." in t) and len(t) >= 6
        ]
        if severe_ungrounded:
            for ut in severe_ungrounded[:3]:
                unverified.append(
                    f"Ungrounded code identifier '{ut}' not found in retrieved repository evidence."
                )

    total_checked = len(candidate_code_tokens) + verified_citations
    if total_checked > 0:
        grounding_score = (grounded_count + verified_citations) / float(total_checked)
    else:
        grounding_score = 1.0

    passed = len(unverified) == 0

    # Deterministic Fast-Pass:
    # Skip the secondary LLM Judge call when:
    # 1) Zero deterministic errors occurred (passed is True), AND
    # 2) At least 1 explicit repository citation (Commit SHA or PR/Issue) was verified in the evidence, AND
    # 3) Structural code identifier grounding score >= 0.70.
    fast_pass = bool(passed and verified_citations >= 1 and grounding_score >= 0.70)

    return DeterministicVerificationResult(
        passed=passed,
        unverified_claims=unverified,
        verified_citation_count=verified_citations,
        grounding_score=round(grounding_score, 4),
        fast_pass=fast_pass,
    )


# ---------------------------------------------------------------------------
# STAGE 4: Post-Synthesis Provenance & Citation Completeness Guardrail
# ---------------------------------------------------------------------------

def apply_post_synthesis_guardrail(
    final_answer: str,
    sorted_chunks: list[dict[str, Any]],
    evidence_text: str,
    repo_id: str | None = None,
) -> str:
    """Stage 4: Deterministically verify the final synthesized response in `synthesize_node`
    against retrieved evidence + JIT diffs + SQLite, stripping/disclaiming post-draft
    hallucinations and ensuring top-ranked PR/Issue provenance links are never dropped.
    """
    if not final_answer or not final_answer.strip():
        return final_answer

    evidence_index = build_evidence_index(
        sorted_chunks=sorted_chunks,
        evidence_text=evidence_text,
        repo_id=repo_id,
        candidate_text=final_answer,
    )

    # 1. Check for post-draft hallucinated SHAs or PRs/Issues introduced during synthesis
    hallucinated_refs: list[str] = []
    for pr_num in extract_cited_prs_issues(final_answer):
        if not evidence_index.is_known_pr_or_issue(pr_num):
            hallucinated_refs.append(f"#{pr_num}")

    for sha in extract_cited_shas(final_answer, evidence_index.known_shas):
        if not evidence_index.is_known_sha(sha):
            valid_in_db = False
            try:
                with get_session_context() as session:
                    valid_in_db = is_valid_commit_sha(session, sha.lower(), repo_id=repo_id)
            except Exception:
                valid_in_db = False
            if not valid_in_db:
                hallucinated_refs.append(f"commit `{sha}`")

    # 2. Ensure verified linked PRs/Issues from the top-3 chunks are cited
    top_verified_links: list[str] = []
    seen_links: set[str] = set()
    for c in sorted_chunks[:3]:
        if not isinstance(c, dict):
            continue
        for rel in c.get("related_ids") or []:
            rel_s = str(rel).strip()
            m_num = re.search(r"(\d+)", rel_s)
            if m_num:
                num_s = m_num.group(1)
                if num_s not in seen_links:
                    seen_links.add(num_s)
                    label = f"PR #{num_s}" if "pr" in rel_s.lower() else (
                        f"Issue #{num_s}" if "issue" in rel_s.lower() else f"#{num_s}"
                    )
                    top_verified_links.append((num_s, label, str(c.get("source_id", ""))[:7]))

    missing_top_links = [
        (num_s, label, sha_ref)
        for num_s, label, sha_ref in top_verified_links
        if not re.search(rf"\b(?:pr|issue|#)\s*#?{re.escape(num_s)}\b", final_answer, re.IGNORECASE)
    ]

    updated_answer = final_answer
    if missing_top_links and "### Historical PRs & Linked Issues" not in updated_answer:
        ref_lines = ["\n\n### Historical PRs & Linked Issues"]
        for _, label, sha_ref in missing_top_links:
            if sha_ref:
                ref_lines.append(f"- **{label}** (linked to commit `{sha_ref}`)")
            else:
                ref_lines.append(f"- **{label}**")
        updated_answer += "\n".join(ref_lines)

    if hallucinated_refs:
        refs_str = ", ".join(hallucinated_refs)
        if "could not be conclusively verified" not in updated_answer:
            updated_answer += (
                f"\n\n> \u26a0\ufe0f *Note: Provenance guardrail flagged unverified reference(s) "
                f"({refs_str}) not present in retrieved repository evidence.*"
            )

    return updated_answer
