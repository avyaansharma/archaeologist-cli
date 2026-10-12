from typing import Any, TypedDict


class AgentState(TypedDict):
    question: str
    repo_id: str | None
    sub_questions: list[str]
    current_sub_question_index: int
    search_queries: list[str]
    retrieved_chunks: list[dict[str, Any]]
    evidence_by_chunk_id: dict[str, dict[str, Any]]
    draft_answer: str | None
    verification_passed: bool
    unverified_claims: list[str]
    judge_unavailable: bool | None
    retry_count: int
    response: str | None
    _current_plan: dict[str, Any] | None
