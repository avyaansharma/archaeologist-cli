import json
import os
import re
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

os.environ["DATABASE_URL"] = os.getenv("FLASK_DATABASE_URL", "sqlite:///./eval/data/flask.db")
os.environ["BM25_INDEX_PATH"] = os.getenv("FLASK_BM25_PATH", os.path.abspath("eval/data/flask_bm25.bin"))

FLASK_QA_PAIRS = [
    {
        "id": 1,
        "question": "How does Flask handle request context and application context isolation using ContextVar / LocalProxy in src/flask/globals.py and src/flask/ctx.py?",
        "expected_answer": "Flask uses ContextVar inside globals.py (_cv_app and _cv_request) to isolate AppContext and RequestContext instances per thread or async task. Access to current_app, g, request, and session is proxied through LocalProxy instances that dynamically resolve the active context.",
        "expected_refs": ["globals.py", "ctx.py", "LocalProxy", "ContextVar"],
        "propositions": [
            "Flask uses ContextVar inside globals.py to isolate AppContext or RequestContext",
            "Access to current_app, g, request, or session is proxied through LocalProxy instances",
            "LocalProxy dynamically resolves the active context for the current execution thread or task"
        ]
    },
    {
        "id": 2,
        "question": "How does Flask's Blueprint in src/flask/blueprints.py defer route and view function registration until application initialization via Blueprint.record() and BlueprintSetupState?",
        "expected_answer": "Blueprint records route definitions and setup callbacks via Blueprint.record() and BlueprintSetupState. When app.register_blueprint() is called, Flask iterates through all recorded state functions and executes them against the application instance.",
        "expected_refs": ["src/flask/blueprints.py", "register_blueprint", "record", "BlueprintSetupState"],
        "propositions": [
            "Blueprint defers route and view function registration until application initialization",
            "Blueprint records route definitions and setup callbacks via Blueprint.record() or BlueprintSetupState",
            "When app.register_blueprint() is called, Flask iterates through recorded functions and executes them against the app instance"
        ]
    },
    {
        "id": 3,
        "question": "How does Flask process error handling hierarchy between app-level error handlers and blueprint-level error handlers in src/flask/app.py via handle_user_exception?",
        "expected_answer": "Flask maintains app.error_handler_spec dictionary where global errorhandlers use None as key and blueprint errorhandlers use the blueprint name. When handle_user_exception handles an error, it checks blueprint-specific handlers first before falling back to global handlers.",
        "expected_refs": ["src/flask/app.py", "handle_user_exception", "error_handler_spec", "errorhandler"],
        "propositions": [
            "Flask maintains app.error_handler_spec dictionary mapping blueprint names and None for app-level handlers",
            "When handle_user_exception handles an error, it prioritizes blueprint-specific handlers first before falling back to global handlers",
            "Error handlers are resolved by HTTP status code and exception class inheritance hierarchy (MRO)"
        ]
    },

    {
        "id": 4,
        "question": "How does Flask integrate with Click in src/flask/cli.py to create CLI command groups and locate application instances?",
        "expected_answer": "FlaskGroup and AppGroup in src/flask/cli.py extend click.Group. ScriptInfo uses find_best_app() to locate and import the target Flask application instance before executing Click CLI commands.",
        "expected_refs": ["cli.py", "FlaskGroup", "AppGroup", "ScriptInfo"],
        "propositions": [
            "Flask integrates with Click through AppGroup and FlaskGroup which extend click.Group",
            "ScriptInfo is used to discover and import the Flask application instance",
            "AppGroup and FlaskGroup manage application context injection for CLI commands"
        ]
    },
    {
        "id": 5,
        "question": "How does SecureCookieSessionInterface in src/flask/sessions.py sign and validate session cookie data against tampering?",
        "expected_answer": "SecureCookieSessionInterface uses itsdangerous.URLSafeTimedSerializer with the application secret_key and cookie-session salt to serialize, cryptographic sign, and verify session cookie data.",
        "expected_refs": ["sessions.py", "SecureCookieSessionInterface", "URLSafeTimedSerializer", "secret_key"],
        "propositions": [
            "SecureCookieSessionInterface uses itsdangerous URLSafeTimedSerializer to sign and verify cookie session data",
            "Signing is keyed with the application SECRET_KEY and cookie salt",
            "Tampered or invalid signatures raise BadSignature and result in an empty session"
        ]
    },
    {
        "id": 6,
        "question": "How does full_dispatch_request in src/flask/app.py orchestrate the lifecycle of request hooks, dispatching, and response processing?",
        "expected_answer": "full_dispatch_request runs preprocess_request (before_request hooks) first. If no hook returns a response, it dispatches to dispatch_request (view function), converts the return value via make_response, and passes the response through process_response (after_request hooks).",
        "expected_refs": ["app.py", "full_dispatch_request", "preprocess_request", "dispatch_request", "process_response"],
        "propositions": [
            "full_dispatch_request orchestrates request preprocessing, dispatching, and response processing",
            "It executes preprocess_request (before_request hooks) before dispatching",
            "If pre-processing does not short-circuit, it delegates to dispatch_request to execute the view function",
            "The response is finalized and processed through process_response (after_request hooks)"
        ]
    }
]

RESULTS_PATH = os.path.join("eval", "flask_results.json")

def run_flask_eval():
    from archaeologist.mcp_server.tools import ask_tool
    from archaeologist.utils.gemini_client import GeminiClientWrapper
    from eval.metrics import evaluate_atomic_propositions, compute_citation_metrics, compute_rouge_l, compute_calibrated_grounded_accuracy
    
    gemini = GeminiClientWrapper()
    scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    entailment_scores = []
    rouge_l_scores = []
    detailed_results = []

    print("\n" + "=" * 90, flush=True)
    print("RUNNING BENCHMARK EVALUATION ON FAMOUS OPEN-SOURCE REPO: pallets/flask (repo_id='flask')", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Question':<45} | {'Grounded Acc':<12} | {'Fact Entail':<11} | {'Citation F1':<11} | {'ROUGE-L':<8}", flush=True)
    print("-" * 90, flush=True)

    for item in FLASK_QA_PAIRS:
        q = item["question"]
        expected = item["expected_answer"]
        expected_refs = item["expected_refs"]
        propositions = item.get("propositions", [])

        # Run multi-hop reasoning agent with repo_id scoping
        generated = ask_tool(q, repo_id="flask")

        # 1. Proposition Entailment Rate
        prop_res = evaluate_atomic_propositions(generated, propositions, gemini=gemini)
        entail_rate = prop_res["entailment_rate"]
        entailment_scores.append(entail_rate)

        # 2. Citation Metrics
        cit_metrics = compute_citation_metrics(generated, expected_refs)
        citation_prec = cit_metrics["precision"]
        citation_rec = cit_metrics["recall"]
        citation_f1 = cit_metrics["f1"]
        precision_scores.append(citation_prec)
        recall_scores.append(citation_rec)
        f1_scores.append(citation_f1)

        # 3. Lexical ROUGE-L
        rouge_res = compute_rouge_l(generated, expected)
        rouge_l_scores.append(rouge_res["f1"])

        # 4. Calibrated Grounded Accuracy (70% Fact Entailment + 30% Citation F1)
        acc_score = compute_calibrated_grounded_accuracy(entail_rate, citation_f1, has_citations=bool(expected_refs))
        scores.append(acc_score)

        detailed_results.append({
            "id": item["id"],
            "question": q,
            "expected_answer": expected,
            "generated_answer": generated,
            "grounded_accuracy": acc_score,
            "fact_entailment_rate": entail_rate,
            "citation_precision": citation_prec,
            "citation_recall": citation_rec,
            "citation_f1": citation_f1,
            "rouge_l_f1": rouge_res["f1"],
            "proposition_evaluations": prop_res.get("evaluations", [])
        })

        q_trunc = q[:42] + "..." if len(q) > 45 else q
        print(f"{q_trunc:<45} | {acc_score:<12.2%} | {entail_rate:<11.2%} | {citation_f1:<11.2%} | {rouge_res['f1']:<8.2%}", flush=True)

    avg_acc = sum(scores) / len(scores) if scores else 0.0
    avg_prec = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_rec = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_entail = sum(entailment_scores) / len(entailment_scores) if entailment_scores else 0.0
    avg_rouge = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0

    print("-" * 90, flush=True)
    print(f"{'AVERAGE SUMMARY (pallets/flask Benchmark)':<45} | {avg_acc:<12.2%} | {avg_entail:<11.2%} | {avg_f1:<11.2%} | {avg_rouge:<8.2%}", flush=True)
    print("-" * 90, flush=True)

    summary = {
        "repository": "pallets/flask",
        "repo_id": "flask",
        "total_questions": len(FLASK_QA_PAIRS),
        "average_grounded_accuracy": avg_acc,
        "average_fact_entailment_rate": avg_entail,
        "average_citation_precision": avg_prec,
        "average_citation_recall": avg_rec,
        "average_citation_f1": avg_f1,
        "average_rouge_l_f1": avg_rouge,
        "results": detailed_results
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved evaluation benchmark results to {RESULTS_PATH}", flush=True)

if __name__ == "__main__":
    run_flask_eval()
