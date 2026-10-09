import re
import math
from typing import List, Dict, Any, Optional
from archaeologist.retrieval.embedder import Embedder
from archaeologist.utils.gemini_client import GeminiClientWrapper

def compute_lcs_length(x_tokens: List[str], y_tokens: List[str]) -> int:
    """Computes the Longest Common Subsequence length between two token lists."""
    m = len(x_tokens)
    n = len(y_tokens)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x_tokens[i - 1] == y_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def compute_rouge_l(candidate: str, reference: str) -> Dict[str, float]:
    """Computes standard ROUGE-L precision, recall, and F1 score."""
    cand_tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', candidate.lower())
    ref_tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', reference.lower())
    
    if not cand_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    lcs_len = compute_lcs_length(cand_tokens, ref_tokens)
    prec = float(lcs_len) / len(cand_tokens) if cand_tokens else 0.0
    rec = float(lcs_len) / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    
    return {"precision": prec, "recall": rec, "f1": f1}

def compute_semantic_cosine_similarity(text1: str, text2: str, embedder: Optional[Embedder] = None) -> float:
    """Computes cosine similarity between embeddings of text1 and text2."""
    if not text1.strip() or not text2.strip():
        return 0.0
    
    emb = embedder or Embedder()
    vectors, flags = emb.embed_texts([text1, text2], return_success_flags=True)
    if not vectors or len(vectors) < 2 or not flags[0] or not flags[1]:
        return 0.0
    
    v1, v2 = vectors[0], vectors[1]
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))

def compute_citation_metrics(generated: str, expected_refs: List[str]) -> Dict[str, float]:
    """Computes Citation Recall, Precision, and F1 based on canonicalized references."""
    if not expected_refs:
        return {"recall": 1.0, "precision": 1.0, "f1": 1.0}
    
    gen_lower = generated.lower()
    
    # 1. Canonicalize extracted SHA hashes (deduplicating short prefixes of full SHAs)
    raw_shas = set(re.findall(r'\b[0-9a-fA-F]{7,40}\b', generated))
    canonical_shas = set()
    sorted_shas = sorted(raw_shas, key=len, reverse=True)
    for s in sorted_shas:
        s_lower = s.lower()
        if not any(s_lower != existing and existing.startswith(s_lower) for existing in canonical_shas):
            canonical_shas.add(s_lower)
            
    # 2. Extract PR numbers and Issue numbers
    cited_prs = {f"pr#{m}" for m in re.findall(r'PR\s*#?(\d+)', generated, re.IGNORECASE)}
    cited_issues = {f"issue#{m}" for m in re.findall(r'Issue\s*#?(\d+)', generated, re.IGNORECASE)}
    
    all_cited_entities = canonical_shas | cited_prs | cited_issues
    
    # 3. Check expected references recall
    matched_expected = 0
    for exp in expected_refs:
        exp_clean = exp.lower().strip()
        if any(exp_clean in entity or entity in exp_clean for entity in all_cited_entities) or exp_clean in gen_lower:
            matched_expected += 1
            
    recall = float(matched_expected) / len(expected_refs) if expected_refs else 1.0
    
    # 4. Precision: proportion of cited entities that match expected references
    relevant_cited = 0
    for entity in all_cited_entities:
        if any(exp.lower().strip() in entity or entity in exp.lower().strip() for exp in expected_refs):
            relevant_cited += 1
            
    if not all_cited_entities:
        precision = 0.0
    elif relevant_cited > 0:
        precision = min(1.0, float(relevant_cited) / len(all_cited_entities))
    else:
        precision = 1.0 if recall >= 1.0 else 0.0
        
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1
    }


def evaluate_atomic_propositions(generated: str, propositions: List[str], gemini: Optional[GeminiClientWrapper] = None) -> Dict[str, Any]:
    """Evaluates whether each atomic proposition from the expected ground-truth is entailed by the generated answer."""
    if not propositions:
        return {"entailment_rate": 1.0, "results": []}
    
    client = gemini or GeminiClientWrapper()
    prompt = (
        "You are an objective mathematical truth evaluator.\n"
        "Given the following Generated Answer, determine whether each atomic proposition is explicitly supported (ENTAILED = true) or absent/refuted (ENTAILED = false).\n\n"
        f"Generated Answer:\n{generated}\n\n"
        f"Atomic Propositions to verify:\n" + "\n".join(f"- [{i}] {p}" for i, p in enumerate(propositions)) + "\n\n"
        "Return a JSON object with field 'evaluations' as a list of boolean values [true, false, ...] matching the order of propositions."
    )
    
    try:
        res = client.generate_json(prompt=prompt, model="gemini-3.5-flash", temperature=0.0)
        evals = res.get("evaluations", [])
        if isinstance(evals, list) and len(evals) == len(propositions):
            true_count = sum(1 for e in evals if bool(e))
            rate = float(true_count) / len(propositions)
            return {
                "entailment_rate": rate,
                "evaluations": evals,
                "passed_count": true_count,
                "total_propositions": len(propositions)
            }
    except Exception:
        pass
    
    # Fallback to simple keyword check if LLM judge fails
    gen_lower = generated.lower()
    fallback_evals = []
    for p in propositions:
        p_keywords = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_]{3,}\b', p)]
        match_count = sum(1 for w in p_keywords if w in gen_lower)
        fallback_evals.append(match_count >= max(1, len(p_keywords) // 2))
    
    true_count = sum(1 for e in fallback_evals if e)
    rate = float(true_count) / len(propositions)
    return {
        "entailment_rate": rate,
        "evaluations": fallback_evals,
        "passed_count": true_count,
        "total_propositions": len(propositions)
    }

def compute_calibrated_grounded_accuracy(entailment_rate: float, citation_f1: float, has_citations: bool = True) -> float:
    """Computes calibrated Grounded Accuracy score combining 70% Fact Entailment and 30% Citation F1."""
    if not has_citations:
        return entailment_rate
    return (0.70 * entailment_rate) + (0.30 * citation_f1)

