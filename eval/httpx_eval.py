import json
import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

HTTPX_QA_PAIRS = [
    {
        "id": 1,
        "question": "How does httpx bridge async and sync HTTP transports between HTTPTransport and AsyncHTTPTransport using httpcore in httpx/_transports/default.py?",
        "expected_answer": "httpx delegates connection management and HTTP/1.1 or HTTP/2 wire protocols to httpcore. HTTPTransport wraps httpcore.HTTPConnectionPool for synchronous I/O, while AsyncHTTPTransport wraps httpcore.AsyncHTTPConnectionPool for asynchronous I/O using anyio.",
        "expected_refs": ["_transports/default.py", "HTTPTransport", "AsyncHTTPTransport", "httpcore"]
    },
    {
        "id": 2,
        "question": "How does Client.send() in httpx/_client.py process request building, auth flows, redirect handling, and response streaming?",
        "expected_answer": "Client.send() constructs a Request object, applies authentication via Auth or generator flows, delegates transport execution via _send_handling_auth and _send_single_request, handles HTTP redirects dynamically up to max_redirects, and returns a Response object.",
        "expected_refs": ["_client.py", "Client", "send", "_send_handling_auth", "max_redirects"]
    },
    {
        "id": 3,
        "question": "How does Timeout and Limits configuration in httpx/_config.py enforce connect, read, write, and pool timeouts across HTTP requests?",
        "expected_answer": "httpx defines granular Timeout instances with connect, read, write, and pool attributes. Limits controls max_connections and max_keepalive_connections, which are passed directly to the underlying httpcore connection pool.",
        "expected_refs": ["_config.py", "Timeout", "Limits", "max_connections", "max_keepalive_connections"]
    },
    {
        "id": 4,
        "question": "How does create_ssl_context in httpx/_config.py configure SSL/TLS verification, CA certificates, and HTTP/2 ALPN negotiation?",
        "expected_answer": "create_ssl_context uses ssl.create_default_context() with verify and cert parameters. It configures ALPN protocols ('h2', 'http/1.1') when HTTP/2 is enabled and loads client certificates via load_cert_chain.",
        "expected_refs": ["_config.py", "create_ssl_context", "verify", "ALPN", "load_cert_chain"]
    }
]

RESULTS_PATH = os.path.join("eval", "httpx_results.json")

def run_httpx_eval():
    from archaeologist.mcp_server.tools import ask_tool
    from archaeologist.utils.gemini_client import GeminiClientWrapper
    
    gemini = GeminiClientWrapper()
    scores = []
    precision_scores = []
    detailed_results = []

    print("\n" + "=" * 90, flush=True)
    print("RUNNING BENCHMARK EVALUATION ON FAMOUS OPEN-SOURCE REPO: encode/httpx", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Question':<55} | {'Grounded Acc':<12} | {'Citation Prec':<12}", flush=True)
    print("-" * 90, flush=True)

    for item in HTTPX_QA_PAIRS:
        q = item["question"]
        expected = item["expected_answer"]
        expected_refs = item["expected_refs"]

        # Run multi-hop reasoning agent
        generated = ask_tool(q)

        # Judge Grounded Accuracy using Gemini 3.5 Flash
        prompt = (
            "You are an expert evaluator scoring the accuracy of an agentic code analysis system.\n"
            "Evaluate whether the generated answer accurately conveys the key architectural facts of the expected answer for the httpx codebase.\n\n"
            f"Question: {q}\n"
            f"Expected Answer: {expected}\n"
            f"Generated Answer: {generated}\n\n"
            "Score the answer on a scale from 0 to 10 (where 10 is perfectly accurate and 0 is wrong/hallucinated).\n"
            "Respond ONLY with a single JSON object with fields:\n"
            "- score: integer between 0 and 10\n"
            "- reasoning: string explaining your score\n\n"
            "Output ONLY the JSON object. Do not include markdown code block formatting."
        )

        acc_score = 0.0
        for attempt in range(3):
            try:
                eval_res = gemini.generate_json(prompt=prompt, model="gemini-3.5-flash")
                acc_score = float(eval_res.get("score", 0)) / 10.0
                break
            except Exception as e:
                print(f"Error in judge evaluation (attempt {attempt + 1}/3): {e}", flush=True)
                import time
                time.sleep(3.0)

        scores.append(acc_score)

        # Calculate Citation Precision
        gen_lower = generated.lower()
        matches = sum(1 for ref in expected_refs if ref.lower() in gen_lower)
        prec_score = float(matches) / len(expected_refs) if expected_refs else 0.0
        precision_scores.append(prec_score)

        detailed_results.append({
            "id": item["id"],
            "question": q,
            "expected_answer": expected,
            "generated_answer": generated,
            "grounded_accuracy": acc_score,
            "citation_precision": prec_score
        })

        q_trunc = q[:52] + "..." if len(q) > 55 else q
        print(f"{q_trunc:<55} | {acc_score:<12.2%} | {prec_score:<12.2%}", flush=True)

    avg_acc = sum(scores) / len(scores) if scores else 0.0
    avg_prec = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

    print("-" * 90, flush=True)
    print(f"{'AVERAGE SUMMARY (encode/httpx Benchmark)':<55} | {avg_acc:<12.2%} | {avg_prec:<12.2%}", flush=True)
    print("-" * 90, flush=True)

    summary = {
        "repository": "encode/httpx",
        "total_questions": len(HTTPX_QA_PAIRS),
        "average_grounded_accuracy": avg_acc,
        "average_citation_precision": avg_prec,
        "results": detailed_results
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved evaluation benchmark results to {RESULTS_PATH}", flush=True)

if __name__ == "__main__":
    run_httpx_eval()
