import pytest
from archaeologist.retrieval.fusion import reciprocal_rank_fusion

def test_reciprocal_rank_fusion():
    # Dense results from Qdrant
    dense_hits = [
        {"id": "chunk_1", "score": 0.9, "payload": {"source_type": "commit", "source_id": "sha_1", "text": "auth fix", "file_paths": ["src/auth.py"], "related_ids": [], "is_reverted": False}},
        {"id": "chunk_2", "score": 0.8, "payload": {"source_type": "commit", "source_id": "sha_2", "text": "retry logic", "file_paths": ["src/fetch.py"], "related_ids": [], "is_reverted": False}},
    ]
    
    # Sparse results from BM25
    sparse_hits = [
        {"score": 5.0, "chunk": {"id": "chunk_2", "source_type": "commit", "source_id": "sha_2", "text": "retry logic", "file_paths": ["src/fetch.py"], "related_ids": [], "is_reverted": False}},
        {"score": 4.5, "chunk": {"id": "chunk_3", "source_type": "commit", "source_id": "sha_3", "text": "other refactor", "file_paths": ["src/main.py"], "related_ids": [], "is_reverted": False}},
    ]
    
    # Run RRF
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, limit=10)
    
    assert len(fused) == 3
    # chunk_2 should be ranked 1st because it is present in both (rank 2 in dense, rank 1 in sparse)
    # RRF score chunk_2 = 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = 0.032522
    # RRF score chunk_1 = 1/(60+1) = 0.016393
    # RRF score chunk_3 = 1/(60+2) = 0.016129
    
    assert fused[0]["id"] == "chunk_2"
    assert fused[1]["id"] == "chunk_1"
    assert fused[2]["id"] == "chunk_3"
