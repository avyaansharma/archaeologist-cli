import json
import os
import re
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

os.environ["DATABASE_URL"] = os.getenv("REQUESTS_DATABASE_URL", "sqlite:///./eval/data/requests.db")
os.environ["BM25_INDEX_PATH"] = os.getenv("REQUESTS_BM25_PATH", os.path.abspath("eval/data/requests_bm25.bin"))

REQUESTS_QA_PAIRS = [
    {
        "id": 1,
        "question": "How does Session.send() in requests/sessions.py handle request preparation, redirect history tracking, connection pool adapter dispatching, and cookie persistence?",
        "expected_answer": "Session.send() dispatches the prepared request to the registered adapter (HTTPAdapter), manages request redirects using resolve_redirects(), records redirect history in response.history, and merges response cookies back into Session.cookies.",
        "expected_refs": ["sessions.py", "Session", "send", "HTTPAdapter", "resolve_redirects", "cookies"],
        "propositions": [
            "Session.send() dispatches the prepared request to the registered transport adapter (e.g. HTTPAdapter)",
            "It manages redirects using resolve_redirects and tracks the chain in response.history",
            "It extracts and persists cookies back into the session's cookie jar"
        ]
    },
    {
        "id": 2,
        "question": "How does HTTPAdapter in requests/adapters.py configure urllib3 connection pooling, max_retries, and SSL verification?",
        "expected_answer": "HTTPAdapter initializes urllib3 PoolManager via init_poolmanager(), sets max_retries on the connection manager, and injects SSL verification context (cert_reqs, ca_certs, ssl_version) into pool connections during request transmission.",
        "expected_refs": ["adapters.py", "HTTPAdapter", "init_poolmanager", "PoolManager", "max_retries"],
        "propositions": [
            "HTTPAdapter configures urllib3 connection pooling via init_poolmanager",
            "It sets max_retries converting integer values to urllib3 Retry objects",
            "It configures SSL/TLS verification context attributes (cert_verify/cert_reqs) during transmission"
        ]
    },
    {
        "id": 3,
        "question": "How does CaseInsensitiveDict in requests/structures.py enforce lower-case key normalization and preserve original key casing during iteration?",
        "expected_answer": "CaseInsensitiveDict maps lower-cased keys to (original_key, value) tuples inside a private _store dict, ensuring case-insensitive lookups while preserving original key casing when iterating over keys or items.",
        "expected_refs": ["structures.py", "CaseInsensitiveDict", "_store"],
        "propositions": [
            "CaseInsensitiveDict maps lower-cased keys to (original_key, value) tuples in an internal _store dict",
            "Lookups normalize keys to lowercase for case-insensitivity",
            "Iteration over keys/items preserves the original casing of the keys"
        ]
    },
    {
        "id": 4,
        "question": "How does Response.raise_for_status() in requests/models.py evaluate HTTP status code ranges to raise HTTPError exceptions?",
        "expected_answer": "raise_for_status() checks response.status_code. If 400 <= status_code < 500, it raises a ClientError or HTTPError; if 500 <= status_code < 600, it raises a ServerError or HTTPError, formatted with response URL and reason.",
        "expected_refs": ["models.py", "Response", "raise_for_status", "HTTPError", "status_code"],
        "propositions": [
            "raise_for_status() evaluates the response HTTP status_code",
            "For 4xx status codes (400-499) it formats a client error message and raises HTTPError",
            "For 5xx status codes (500-599) it formats a server error message and raises HTTPError"
        ]
    }
]

RESULTS_PATH = os.path.join("eval", "requests_results.json")

def run_requests_eval():
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
    print("RUNNING BENCHMARK EVALUATION ON FAMOUS OPEN-SOURCE REPO: psf/requests (repo_id='requests')", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Question':<45} | {'Grounded Acc':<12} | {'Fact Entail':<11} | {'Citation F1':<11} | {'ROUGE-L':<8}", flush=True)
    print("-" * 90, flush=True)

    for item in REQUESTS_QA_PAIRS:
        q = item["question"]
        expected = item["expected_answer"]
        expected_refs = item["expected_refs"]
        propositions = item.get("propositions", [])

        # Run multi-hop reasoning agent with repo_id scoping
        generated = ask_tool(q, repo_id="requests")

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
    print(f"{'AVERAGE SUMMARY (psf/requests Benchmark)':<45} | {avg_acc:<12.2%} | {avg_entail:<11.2%} | {avg_f1:<11.2%} | {avg_rouge:<8.2%}", flush=True)
    print("-" * 90, flush=True)

    summary = {
        "repository": "psf/requests",
        "repo_id": "requests",
        "total_questions": len(REQUESTS_QA_PAIRS),
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
    run_requests_eval()
