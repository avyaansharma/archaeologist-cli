import sys
from langgraph.graph import StateGraph, END
from archaeologist.agent.state import AgentState
from archaeologist.agent.nodes.decompose import decompose_node
from archaeologist.agent.nodes.plan import plan_node
from archaeologist.agent.nodes.search import search_node
from archaeologist.agent.nodes.follow_links import follow_links_node
from archaeologist.agent.nodes.verify import verify_node
from archaeologist.agent.nodes.synthesize import synthesize_node

def advance_sub_question_node(state: AgentState) -> dict:
    idx = state.get("current_sub_question_index", 0) + 1
    return {"current_sub_question_index": idx}

def increment_retry_node(state: AgentState) -> dict:
    retries = state.get("retry_count", 0)
    print(f"Agent: Verification failed. Incrementing retry count to {retries + 1} targeting unverified claims without discarding completed sub-questions...", file=sys.stderr)
    return {
        "retry_count": retries + 1,
        "current_sub_question_index": state.get("current_sub_question_index", 0),
        "draft_answer": None,
        "verification_passed": False
    }


# Create the graph workflow
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("decompose", decompose_node)
workflow.add_node("plan", plan_node)
workflow.add_node("search", search_node)
workflow.add_node("follow_links", follow_links_node)
workflow.add_node("advance_sub_question", advance_sub_question_node)
workflow.add_node("verify", verify_node)
workflow.add_node("increment_retry", increment_retry_node)
workflow.add_node("synthesize", synthesize_node)

# Set entry point
workflow.set_entry_point("decompose")

# Transitions
workflow.add_edge("decompose", "plan")
workflow.add_edge("plan", "search")
workflow.add_edge("search", "follow_links")
workflow.add_edge("follow_links", "advance_sub_question")

# Sub-question loop router (§1.1 Fix)
def sub_question_router(state: AgentState):
    idx = state.get("current_sub_question_index", 0)
    sub_qs = state.get("sub_questions", [])
    if idx < len(sub_qs):
        return "plan"
    return "verify"

workflow.add_conditional_edges(
    "advance_sub_question",
    sub_question_router,
    {
        "plan": "plan",
        "verify": "verify"
    }
)

# Verification router (§1.2 Fix & Audit R4)
def verification_router(state: AgentState):
    if state.get("verification_passed", True):
        return "synthesize"
    
    # Circuit-break retries if judge was unavailable (no API key or judge fatal error)
    unverified = state.get("unverified_claims", [])
    if any("no gemini api key" in str(u).lower() or "judge unavailable" in str(u).lower() for u in unverified):
        print("Agent: Verification judge unavailable (missing key/offline). Skipping retries and proceeding directly to synthesis.", file=sys.stderr)
        return "synthesize"

    retries = state.get("retry_count", 0)
    if retries >= 2:
        print("Agent: Verification failed after maximum retries. Proceeding to synthesis.", file=sys.stderr)
        return "synthesize"
        
    return "retry"

workflow.add_conditional_edges(
    "verify",
    verification_router,
    {
        "synthesize": "synthesize",
        "retry": "increment_retry"
    }
)

workflow.add_edge("increment_retry", "plan")
workflow.add_edge("synthesize", END)

# Compile workflow
agent_graph = workflow.compile()
