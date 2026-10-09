import sys
from archaeologist.agent.state import AgentState
from archaeologist.utils.gemini_client import GeminiClientWrapper, get_gemini_api_key
from archaeologist.storage.db import get_session_context
from archaeologist.storage.models import find_candidate_forensics

PLAN_PROMPT = """You are a codebase archaeologist planner.
Given the current sub-question, generate 2-3 targeted search queries to query the repository commit logs, pull requests, and issues.
MANDATORY: Keep queries concise (2-5 key terms), focusing on exact symbols, PR/Issue numbers, or core architectural concepts (e.g. "strategy pattern", "factory", "deprecated", "refactor") to maximize search recall.{symbol_context}

Sub-question: {sub_question}

Return a JSON object with this exact structure:
{{
  "search_queries": [
    "query 1",
    "query 2"
  ]
}}"""

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
  ]
}}"""

def plan_node(state: AgentState) -> dict:
    idx = state.get("current_sub_question_index", 0)
    sub_qs = state.get("sub_questions", [])
    current_sub_q = sub_qs[idx] if idx < len(sub_qs) else state["question"]

    retry_count = state.get("retry_count", 0)
    unverified = state.get("unverified_claims", [])
    repo_id = state.get("repo_id")

    print(f"Agent: Planning retrieval using Gemini 3.5 Flash for sub-question: '{current_sub_q}' (retry={retry_count})...", file=sys.stderr)

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
        return {"search_queries": [current_sub_q]}

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
            model="gemini-3.5-flash",
            temperature=temp,
            max_output_tokens=2000
        )
        queries = result.get("search_queries", [current_sub_q])
        return {"search_queries": queries if queries else [current_sub_q]}
    except Exception as e:
        print(f"Error in plan_node: {e}", file=sys.stderr)
        return {"search_queries": [current_sub_q]}
