from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    question: str
    repo_id: Optional[str]
    sub_questions: List[str]
    current_sub_question_index: int
    search_queries: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    evidence_by_chunk_id: Dict[str, Dict[str, Any]]
    draft_answer: Optional[str]
    verification_passed: bool
    unverified_claims: List[str]
    retry_count: int
    response: Optional[str]
