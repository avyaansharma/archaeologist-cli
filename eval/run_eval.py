import json
import os
import re
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from archaeologist.utils.gemini_client import GeminiClientWrapper, get_gemini_api_key

load_dotenv()

DEFAULT_QA_PATH = os.path.join("eval", "dataset", "qa_pairs.jsonl")
UNSEEN_QA_PATH = os.path.join("eval", "dataset", "unseen_eval_pairs.jsonl")
RESULTS_PATH = os.path.join("eval", "results.json")

def load_qa_pairs(dataset_path: Optional[str] = None) -> List[Dict[str, Any]]:
    path = dataset_path or (UNSEEN_QA_PATH if os.path.exists(UNSEEN_QA_PATH) else DEFAULT_QA_PATH)
    pairs = []
    if not os.path.exists(path):
        print(f"QA pairs file not found at {path}.", flush=True)
        return []
        
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs

class LLMJudge:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_gemini_api_key()
        self.gemini = GeminiClientWrapper(api_key=self.api_key) if self.api_key else None

    def evaluate_answer(self, question: str, expected: str, generated: str) -> float:
        """Scores generated answer against expected on a scale of 0.0 to 1.0 using Gemini."""
        if not self.gemini:
            words = [w for w in expected.lower().split() if len(w) > 3]
            match_count = sum(1 for w in words if w in generated.lower())
            return min(1.0, match_count / max(1, len(words) * 0.4))

        prompt = (
            "You are an expert evaluator scoring the accuracy of an agentic Q&A system.\n"
            "Evaluate whether the generated answer accurately conveys the key facts of the expected answer.\n\n"
            f"Question: {question}\n"
            f"Expected Answer: {expected}\n"
            f"Generated Answer: {generated}\n\n"
            "Score the answer on a scale from 0 to 10 (where 10 is perfectly accurate and 0 is completely wrong/hallucinated).\n"
            "Respond ONLY with a single JSON object with fields:\n"
            "- score: integer between 0 and 10\n"
            "- reasoning: string explaining your score\n\n"
            "Output ONLY the JSON object. Do not include markdown code block formatting."
        )

        try:
            res = self.gemini.generate_json(prompt=prompt, model="gemini-3.5-flash")
            return float(res.get("score", 0)) / 10.0
        except Exception as e:
            print(f"Error in Gemini evaluation judge: {e}", flush=True)
            return 0.0

def calculate_citation_precision(generated_answer: str, expected_refs: List[str]) -> float:
    """Calculates fraction of expected references that appear in the generated answer."""
    if not expected_refs:
        return 1.0 if generated_answer and "no history" not in generated_answer.lower() else 0.0
        
    gen_lower = generated_answer.lower()
    matches = sum(1 for ref in expected_refs if ref.lower() in gen_lower)
    return float(matches) / len(expected_refs)

def run_evaluation(dataset_path: Optional[str] = None):
    pairs = load_qa_pairs(dataset_path)
    if not pairs:
        print("No evaluation pairs found to run benchmark.", flush=True)
        return

    from archaeologist.mcp_server.tools import ask_tool
    judge = LLMJudge()

    scores = []
    precision_scores = []
    detailed_results = []
    
    print("\nStarting rigorous evaluation benchmark...", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Question':<50} | {'Grounded Acc':<12} | {'Citation Prec':<12}", flush=True)
    print("-" * 80, flush=True)

    for pair in pairs:
        q = pair["question"]
        expected = pair.get("expected_answer") or pair.get("expected_subject") or ""
        expected_refs = pair.get("expected_refs", [])
        
        # Run agent
        generated = ask_tool(q)
        
        # Grounded Accuracy score (0.0 to 1.0)
        acc_score = judge.evaluate_answer(q, expected, generated)
        scores.append(acc_score)
        
        # Citation Precision score (0.0 to 1.0)
        prec_score = calculate_citation_precision(generated, expected_refs)
        precision_scores.append(prec_score)
        
        detailed_results.append({
            "question": q,
            "expected_answer": expected,
            "generated_answer": generated,
            "grounded_accuracy": acc_score,
            "citation_precision": prec_score
        })
        
        q_trunc = q[:47] + "..." if len(q) > 50 else q
        print(f"{q_trunc:<50} | {acc_score:<12.2%} | {prec_score:<12.2%}", flush=True)

    avg_acc = sum(scores) / len(scores) if scores else 0.0
    avg_prec = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

    print("-" * 80, flush=True)
    print(f"{'AVERAGE SUMMARY':<50} | {avg_acc:<12.2%} | {avg_prec:<12.2%}", flush=True)
    print("-" * 80, flush=True)
    
    # Save evaluation summary to results.json
    summary = {
        "dataset_path": dataset_path or UNSEEN_QA_PATH,
        "total_questions": len(pairs),
        "average_grounded_accuracy": avg_acc,
        "average_citation_precision": avg_prec,
        "results": detailed_results
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation benchmark results to {RESULTS_PATH}", flush=True)

    print(f"Target Grounded Accuracy: >= 75.00% (Result: {avg_acc:.2%})", flush=True)
    print(f"Target Citation Precision: >= 90.00% (Result: {avg_prec:.2%})", flush=True)
    if avg_acc >= 0.75 and avg_prec >= 0.90:
        print("SUCCESS: Both evaluation metrics meet success criteria!", flush=True)
    else:
        print("NOTICE: Benchmark completed with evaluation metrics logged.", flush=True)

def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_evaluation(dataset_path)

if __name__ == "__main__":
    main()
