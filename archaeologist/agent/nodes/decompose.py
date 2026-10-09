import sys
from archaeologist.agent.state import AgentState
from archaeologist.utils.gemini_client import GeminiClientWrapper, get_gemini_api_key

DECOMPOSE_PROMPT = """You are a codebase archaeologist assistant.
Decompose the following causal 'why' question about a software repository into 1-3 specific, searchable sub-questions.
Focus on identifying what specific commit changes, pull requests, issues, architectural decisions, or code modifications would explain the rationale behind this code.

User Question: {question}

Return a JSON object with this exact structure:
{{
  "sub_questions": [
    "sub question 1",
    "sub question 2"
  ]
}}"""

def decompose_node(state: AgentState) -> dict:
    question = state["question"]
    print("Agent: Decomposing question using Gemini 3.5 Flash...", file=sys.stderr)

    api_key = get_gemini_api_key()
    if not api_key:
        print("WARNING: GEMINI_API_KEY / GOOGLE_API_KEY not found. Decompose skipped.", file=sys.stderr)
        return {
            "sub_questions": [question],
            "current_sub_question_index": 0,
            "retrieved_chunks": [],
            "evidence_by_chunk_id": {}
        }

    try:
        client = GeminiClientWrapper(api_key=api_key)
        prompt = DECOMPOSE_PROMPT.format(question=question)
        result = client.generate_json(
            prompt=prompt,
            model="gemini-3.5-flash",
            temperature=0.0,
            max_output_tokens=2000
        )
        sub_qs = result.get("sub_questions", [question])
        return {
            "sub_questions": sub_qs if sub_qs else [question],
            "current_sub_question_index": 0,
            "retrieved_chunks": [],
            "evidence_by_chunk_id": {}
        }
    except Exception as e:
        print(f"Error in decompose_node: {e}", file=sys.stderr)
        return {
            "sub_questions": [question],
            "current_sub_question_index": 0,
            "retrieved_chunks": [],
            "evidence_by_chunk_id": {}
        }

