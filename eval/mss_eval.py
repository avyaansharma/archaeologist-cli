import json
import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

os.environ["DATABASE_URL"] = os.getenv("MSS_DATABASE_URL", "sqlite:///./eval/data/mss.db")
os.environ["BM25_INDEX_PATH"] = os.getenv("MSS_BM25_PATH", os.path.abspath("eval/data/mss_bm25.bin"))

MSS_QA_PAIRS = [
    {
        "id": 1,
        "question": (
            "The Windows capture backend once cached the region being grabbed for performance. "
            "A fix was written in commit 5e5f3ee to correct a bug where two same-size regions at different screen "
            "positions could return a wrong/stale screenshot. That fix was reverted in commit 2d24115 just days "
            "later. Why was it reverted, and what happened to the test that was added alongside it?"
        ),
        "expected_answer": (
            "The original bug (commit 5e5f3ee) was that the Windows GDI backend's region cache "
            "compared only (height, width) when deciding whether to reallocate the cached bitmap, "
            "so two regions of the same size but different (top, left) position incorrectly reused "
            "a stale bitmap sized correctly but positioned wrong. The fix changed the check to "
            "compare the full region dict instead of just height/width, and added a regression "
            "test (test_region_caching) asserting the cache updates when position changes. Three "
            "days later this fix was reverted (commit 2d24115) with the message 'The patch was "
            "indeed a bad idea and fixed nothing' — the author reverted the actual code change but "
            "explicitly kept the added test in place rather than removing it."
        ),
        "expected_evidence": [
            "5e5f3ee",
            "2d24115"
        ],
        "difficulty": "hard",
        "blind_test_notes": (
            "High confidence this is not guessable: requires knowing the exact revert happened, "
            "that it was 3 days after the original fix, that the fix looked correct (passing test) "
            "but 'fixed nothing', and the specific detail that the test was kept despite the code "
            "being reverted. No general GDI/screenshot-caching knowledge would produce this."
        ),
        "propositions": [
            "The original Windows capture region caching fix addressed same-size regions returning stale screenshots",
            "The fix was reverted in commit 2d24115 stating the patch was a bad idea and fixed nothing",
            "The regression test added with the fix was preserved/kept in the test suite rather than removed"
        ]
    },
    {
        "id": 2,
        "question": (
            "In PR #452 (commit 06dc845), this library moved from a single global thread lock protecting all internal state to "
            "per-object locking. What was the stated reason for the change, and which specific "
            "backend still needed to keep a global (not per-object) lock, and why?"
        ),
        "expected_answer": (
            "PR #452 (commit 06dc845) replaced a single global lock with per-object locks, stating "
            "the global lock 'provides too much of a surface for contention and deadlocks.' The "
            "Xlib backend specifically was called out as still needing global locking, because "
            "Xlib itself is not thread-safe and the library wasn't enabling the (partial) "
            "thread-safety features Xlib offers, so Xlib kept its own dedicated lock object "
            "separate from the new per-object scheme used elsewhere."
        ),
        "expected_evidence": [
            "06dc845",
            "pr#452",
            "PR #452"
        ],
        "difficulty": "medium",
        "blind_test_notes": (
            "Medium-high confidence: a model might guess 'global locks cause contention, so they "
            "moved to finer-grained locking' as a generic pattern, but the specific carve-out for "
            "Xlib being non-thread-safe and needing to retain a global lock is a repo-specific "
            "detail unlikely to be guessed correctly."
        ),
        "propositions": [
            "The migration from a single global lock to per-object locking was done to reduce contention and deadlock risk (PR #452)",
            "The Xlib backend (src/mss/linux/xlib.py) specifically retained a global lock",
            "Xlib required a dedicated global lock because underlying Xlib is not thread-safe"
        ]
    },
    {
        "id": 3,
        "question": (
            "In PR #467 (commit 0822b33) and PR #468 (commit 9637209), a fix for a KeyboardInterrupt-related bug during buffer copying in the Linux "
            "capture backend was reworked. What was the original bug, and what specifically changed in the follow-up PR #468?"
        ),
        "expected_answer": (
            "In the XShmGetImage backend, there was a window during buffer copying where a "
            "KeyboardInterrupt (or other async exception) hitting mid-copy would cause cleanup to "
            "raise a different, confusing exception instead of letting the original KeyboardInterrupt "
            "propagate to the user (PR #467, commit 0822b33). The stated goal was not full "
            "correctness under async exceptions, just avoiding a misleading secondary exception. "
            "The follow-up (PR #468, commit 9637209) reworked the fix after the author 'just learned "
            "that memoryviews can be used as context managers' and release their buffers automatically "
            "at the end of a `with` block, replacing the original approach with this simpler pattern."
        ),
        "expected_evidence": [
            "0822b33",
            "9637209",
            "pr#467",
            "pr#468",
            "PR #467",
            "PR #468"
        ],
        "difficulty": "hard",
        "blind_test_notes": (
            "High confidence: the specific detail that the second PR's motivation was the author "
            "personally learning about memoryview context-manager semantics, rather than a bug in "
            "the first fix, is a very specific, personal, undocumented-elsewhere detail that general "
            "knowledge of Python or screenshot libraries would not produce."
        ),
        "propositions": [
            "In the Linux XShmGetImage backend, a KeyboardInterrupt during buffer copying caused a secondary confusing exception during cleanup (PR #467)",
            "The follow-up fix in PR #468 used memoryview objects as context managers",
            "Using memoryviews as context managers ensured buffers are automatically and reliably released at the end of the block"
        ]
    },
    {
        "id": 4,
        "question": (
            "In PR #494 / Issue #486 (commit 8a7bbc2), the library's internal architecture was reworked to use a strategy implementation design with a single "
            "top-level MSS class. What issue motivated this, and what specifically got deprecated as a result?"
        ),
        "expected_answer": (
            "PR #494 (commit 8a7bbc2) moved the library to a strategy-pattern design, citing issue "
            "#486 as the motivation. The change hides platform/capture-strategy differences behind "
            "a single mss.MSS class rather than exposing different implementation classes per "
            "platform, so internal class hierarchies could change freely without breaking user code. "
            "This deprecated both the existing `mss` factory function and the per-platform "
            "`mss.{platform}.MSS` types, with a stated plan to keep the factory function available "
            "for a transition period while deprecated functionality emits DeprecationWarnings."
        ),
        "expected_evidence": [
            "8a7bbc2",
            "pr#494",
            "PR #494",
            "issue#486",
            "Issue #486"
        ],
        "difficulty": "medium",
        "blind_test_notes": (
            "Medium confidence: 'moved to strategy pattern for extensibility' is a somewhat guessable "
            "generic architecture narrative, but the specific linked issue number and the precise "
            "two things deprecated (the factory function AND the per-platform types) are specific "
            "enough that a blind guess is unlikely to get both right."
        ),
        "propositions": [
            "The library unified platform interfaces behind a single top-level mss.MSS class (PR #494, Issue #486)",
            "The redesign hid platform-specific implementations so internal class structures could evolve without breaking user code",
            "It deprecated the legacy mss factory function and the per-platform mss.{platform}.MSS types"
        ]
    },
    {
        "id": 5,
        "question": (
            "In PR #426 (commit 8575606) and PR #431 (commit 712503a), two different low-level backends for capturing screenshots on Linux via X11 "
            "were introduced (XCB getimage and XShmGetImage shared memory). Why was the XShmGetImage backend added over the standard XGetImage backend, and what "
            "performance difference was reported in PR #431?"
        ),
        "expected_answer": (
            "PR #426 (commit 8575606) first added an XCB-based backend named 'getimage' as a "
            "replacement path for the older Xlib implementation, which was explicitly designated "
            "'legacy' status per issue #425 (new functionality like XShmGetImage would not be "
            "added to the Xlib backend going forward). PR #431 (commit 712503a) then added the "
            "XShmGetImage-based backend using the X MIT-SHM extension for shared-memory transfer. "
            "The PR description reports a concrete benchmark from the author's own machine: the new "
            "shared-memory backend achieved 30-34 fps capturing 4k screenshots, versus 11-14 fps for "
            "the existing XGetImage-based approach — roughly a 3x throughput improvement."
        ),
        "expected_evidence": [
            "8575606",
            "712503a",
            "pr#426",
            "pr#431",
            "PR #426",
            "PR #431",
            "issue#425",
            "Issue #425"
        ],
        "difficulty": "hard",
        "blind_test_notes": (
            "High confidence: the specific FPS numbers (30-34 vs 11-14) and the framing that Xlib "
            "was designated legacy 'per issue #425' are precise, sourced facts that general "
            "knowledge of X11/screenshot libraries would not reproduce, even though 'shared memory "
            "is faster than round-tripping image data' is a guessable general principle on its own."
        ),
        "propositions": [
            "The MIT-SHM (XShmGetImage) backend was added in PR #431 / commit 712503a for shared-memory capture",
            "The older Xlib implementation was designated legacy per Issue #425 in favor of XCB / SHM backends",
            "The shared-memory backend reported significant performance gains (approx. 3x improvement / 30-34 fps vs 11-14 fps)"
        ]
    },
]



RESULTS_PATH = os.path.join("eval", "mss_results.json")


def run_mss_eval():
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
    print("RUNNING GROUND-TRUTH CAUSAL ('WHY') BENCHMARK ON: BoboTiG/python-mss (repo_id='mss')", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Question':<45} | {'Grounded Acc':<12} | {'Fact Entail':<11} | {'Citation F1':<11} | {'ROUGE-L':<8}", flush=True)
    print("-" * 90, flush=True)

    for item in MSS_QA_PAIRS:
        q = item["question"]
        expected = item["expected_answer"]
        expected_refs = item["expected_evidence"]
        propositions = item.get("propositions", [])

        # Run multi-hop reasoning agent with repo_id scoping
        generated = ask_tool(q, repo_id="mss")

        # 1. Proposition Entailment Rate
        prop_res = evaluate_atomic_propositions(generated, propositions, gemini=gemini)
        entail_rate = prop_res["entailment_rate"]
        entailment_scores.append(entail_rate)

        # 2. Citation Metrics against real commit SHAs / references
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
            "expected_evidence": expected_refs,
            "difficulty": item["difficulty"],
            "blind_test_notes": item["blind_test_notes"],
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
    print(f"{'AVERAGE SUMMARY (python-mss Ground-Truth Benchmark)':<45} | {avg_acc:<12.2%} | {avg_entail:<11.2%} | {avg_f1:<11.2%} | {avg_rouge:<8.2%}", flush=True)
    print("-" * 90, flush=True)

    summary = {
        "repository": "BoboTiG/python-mss",
        "repo_id": "mss",
        "eval_type": "ground_truth_causal_why",
        "total_questions": len(MSS_QA_PAIRS),
        "average_grounded_accuracy": avg_acc,
        "average_fact_entailment_rate": avg_entail,
        "average_citation_precision": avg_prec,
        "average_citation_recall": avg_rec,
        "average_citation_f1": avg_f1,
        "average_rouge_l_f1": avg_rouge,
        "results": detailed_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved evaluation benchmark results to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    run_mss_eval()
