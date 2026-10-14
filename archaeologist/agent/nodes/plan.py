import sys

from archaeologist.agent.state import AgentState
from archaeologist.storage.db import get_session_context
from archaeologist.storage.models import find_candidate_forensics
from archaeologist.utils.gemini_client import DEFAULT_MODEL, GeminiClientWrapper, get_gemini_api_key

PLAN_PROMPT = """You are a codebase archaeologist planner.
Given the current sub-question, generate 2-3 targeted search queries to query the repository commit logs, pull requests, and issues.
MANDATORY: Keep queries concise (2-5 key terms), focusing on exact symbols, PR/Issue numbers, or core architectural concepts (e.g. "strategy pattern", "factory", "deprecated", "refactor") to maximize search recall.{symbol_context}

Sub-question: {sub_question}

Return a JSON object with this exact structure:
{{
  "search_queries": [
    "query 1",
    "query 2"
  ],
  "file_path": null,
  "source_types": null
}}

Only set "file_path" if the question names a specific file. Only set "source_types" if the question explicitly restricts the source; allowed values: "commit", "pr", "issue", "file", "code". Otherwise leave both null."""

ALLOWED_SOURCE_TYPES = {"commit", "pr", "issue", "file", "code"}


def _normalize_filters(result: dict) -> dict:
    """Validates LLM-provided filters; invalid values are dropped rather than silently emptying retrieval."""
    fp = result.get("file_path")
    fp = fp.strip() if isinstance(fp, str) and fp.strip() else None
    st = result.get("source_types")
    if isinstance(st, str):
        st = [st]
    if isinstance(st, list):
        st = [s.strip().lower() for s in st if isinstance(s, str) and s.strip().lower() in ALLOWED_SOURCE_TYPES]
    else:
        st = None
    return {"file_path": fp, "source_types": st or None}

RETRY_PLAN_PROMPT = """You are a codebase archaeologist planner.
The previous retrieval attempt failed verification because the following specific claims could not be verified from retrieved evidence:
{unverified_claims}{symbol_context}

Sub-question: {sub_question}

Generate 2-3 NEW, distinct, highly targeted search queries specifically designed to locate evidence addressing these unverified claims and answer the sub-question.
MANDATORY: Keep queries concise (2-5 key terms), focusing on specific function names, commit SHAs, file paths, or exact concepts.

Return a JSON object with this exact structure:
{{
  "search_queries": [
    "query 1",
    "query 2"
  ],
  "file_path": null,
  "source_types": null
}}

Only set "file_path" if the question names a specific file. Only set "source_types" if the question explicitly restricts the source; allowed values: "commit", "pr", "issue", "file", "code". Otherwise leave both null."""

def plan_node(state: AgentState) -> dict:
    idx = state.get("current_sub_question_index", 0)
    sub_qs = state.get("sub_questions", [])
    retry_count = state.get("retry_count", 0)
    unverified = state.get("unverified_claims", [])
    repo_id = state.get("repo_id")

    unverified = [str(u.get('claim', u) if isinstance(u, dict) else u) for u in (unverified or []) if u]

    # On retry, target unverified claims while preserving the original question context
    if retry_count > 0 and unverified:
        current_sub_q = f"{state['question']} - Address unverified claims: {'; '.join(unverified)}"
    elif idx < len(sub_qs):
        current_sub_q = sub_qs[idx]
    else:
        current_sub_q = state["question"]

    print(f"Agent: Planning retrieval using Gemini ({DEFAULT_MODEL}) for sub-question: '{current_sub_q}' (retry={retry_count})...", file=sys.stderr)

    # 1. Look up candidate code symbols, PRs, and commit SHAs
    symbol_context = ""
    try:
        with get_session_context() as session:
            candidate_evidence = find_candidate_forensics(session, current_sub_q, repo_id=repo_id)
            if candidate_evidence:
                evidence_str = "\n".join(f"- {e}" for e in candidate_evidence)
                symbol_context = f"\n\nDiscovered verified entities in codebase:\n{evidence_str}\nUse these exact symbols, PR numbers, or commit SHAs in your search queries when relevant."
    except Exception as e:
        print(f"Notice: Candidate forensics lookup skipped: {e}", file=sys.stderr)

    api_key = get_gemini_api_key()
    if not api_key:
        plan_dict = {
            "queries": [current_sub_q],
            "sub_question": current_sub_q,
            "repo_id": repo_id,
            "file_path": None,
            "source_types": None,
            "date_from": None,
            "date_to": None,
        }
        return {"search_queries": [current_sub_q], "_current_plan": plan_dict}

    try:
        client = GeminiClientWrapper(api_key=api_key)
        if retry_count > 0 and unverified:
            claims_str = "\n".join(f"- {c}" for c in unverified)
            prompt = RETRY_PLAN_PROMPT.format(
                sub_question=current_sub_q,
                unverified_claims=claims_str,
                symbol_context=symbol_context
            )
            temp = 0.3
        else:
            prompt = PLAN_PROMPT.format(
                sub_question=current_sub_q,
                symbol_context=symbol_context
            )
            temp = 0.0

        result = client.generate_json(
            prompt=prompt,
            model=DEFAULT_MODEL,
            temperature=temp,
            max_output_tokens=2000
        )
        if not isinstance(result, dict):
            result = {}
        queries = result.get("search_queries") or [current_sub_q]
        if not isinstance(queries, list):
            queries = [str(queries)]
        queries = [str(q) for q in queries if q] or [current_sub_q]
        filters = _normalize_filters(result)
        plan_dict = {
            "queries": queries,
            "sub_question": current_sub_q,
            "repo_id": repo_id,
            "file_path": filters["file_path"],
            "source_types": filters["source_types"],
            "date_from": None,
            "date_to": None,
        }
        return {"search_queries": queries, "_current_plan": plan_dict}
    except Exception as e:
        print(f"Error in plan_node: {e}", file=sys.stderr)
        plan_dict = {
            "queries": [current_sub_q],
            "sub_question": current_sub_q,
            "repo_id": repo_id,
            "file_path": None,
            "source_types": None,
            "date_from": None,
            "date_to": None,
        }
        return {"search_queries": [current_sub_q], "_current_plan": plan_dict}
