import os
import sys
import uuid
import warnings
from typing import List, Dict, Any, Optional
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, Range, PayloadSchemaType
from dotenv import load_dotenv

from archaeologist.storage.paths import get_default_qdrant_path

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

import threading
import atexit

_LOCAL_CLIENTS: Dict[str, QdrantClient] = {}
_CLIENT_LOCK = threading.Lock()

def _cleanup_all_local_clients():
    with _CLIENT_LOCK:
        for p, client in list(_LOCAL_CLIENTS.items()):
            try:
                if hasattr(client, "close"):
                    client.close()
            except Exception:
                pass
        _LOCAL_CLIENTS.clear()

atexit.register(_cleanup_all_local_clients)

class VectorStoreLockedError(RuntimeError):
    """Raised when a write operation is attempted on a locked local Qdrant directory."""
    pass


class VectorStore:
    def __init__(
        self,
        collection_name: str = "repo_history",
        vector_size: Optional[int] = None,
        storage_path: Optional[str] = None,
        writable: bool = False
    ):
        self.collection_name = collection_name
        self.storage_path = storage_path
        self.writable = writable
        self.degraded = False
        self.dimension_mismatch = False
        self.recreated_due_to_dim_change = False

        if vector_size is None:
            from archaeologist.retrieval.embedder import Embedder
            self.vector_size = Embedder().dimension
        else:
            self.vector_size = vector_size
        self.is_in_memory_fallback = False
        
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.is_remote = bool(qdrant_url)
        
        self.client = self._init_client(qdrant_url, qdrant_api_key)

    def _init_client(self, qdrant_url: Optional[str], qdrant_api_key: str) -> QdrantClient:
        """Connects to remote server Qdrant if explicitly configured, otherwise uses local Qdrant."""
        if qdrant_url:
            remote_timeout = float(os.getenv("QDRANT_TIMEOUT", "30.0"))
            try:
                if qdrant_api_key:
                    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=remote_timeout, check_compatibility=False)
                else:
                    client = QdrantClient(url=qdrant_url, timeout=remote_timeout, check_compatibility=False)
                client.get_collections()
                print(f"Connected to Qdrant server at {qdrant_url}", file=sys.stderr)
                self.is_in_memory_fallback = False
                return client
            except Exception as e:
                print(f"Warning: Configured Qdrant server at {qdrant_url} failed ({e}). Falling back to local storage.", file=sys.stderr)

        self.is_in_memory_fallback = False
        raw_path = self.storage_path or get_default_qdrant_path()
        local_path = os.path.abspath(raw_path)
        
        with _CLIENT_LOCK:
            if local_path in _LOCAL_CLIENTS:
                return _LOCAL_CLIENTS[local_path]
            try:
                client = QdrantClient(path=local_path, check_compatibility=False)
                _LOCAL_CLIENTS[local_path] = client
                return client
            except Exception as e:
                if self.writable:
                    raise VectorStoreLockedError(
                        f"Local Qdrant storage at {local_path} is locked by another process (MCP server or UI). "
                        "Stop it before running ingestion to prevent data loss."
                    ) from e
                print(f"Notice: Local Qdrant storage at {local_path} locked by another process ({e}). Operating in degraded read-only mode.", file=sys.stderr)
                self.degraded = True
                self.is_in_memory_fallback = True
                return QdrantClient(":memory:", check_compatibility=False)

    def init_collection(self, recreate: bool = False, force_recreate: bool = False, repo_id: Optional[str] = None):
        """Creates the collection and payload indexes if they do not exist or if vector size mismatches on write."""
        should_recreate = recreate or force_recreate
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if exists:
                if should_recreate:
                    if self.is_remote and repo_id:
                        # On shared remote Qdrant, delete only points for this repo rather than dropping the collection
                        try:
                            self.client.delete(
                                collection_name=self.collection_name,
                                points_selector=Filter(
                                    must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))]
                                )
                            )
                        except Exception as e:
                            print(f"Notice: Remote point purge by repo_id failed ({e}), continuing...", file=sys.stderr)
                    else:
                        if hasattr(self.client, "_client") and hasattr(self.client._client, "collections"):
                            local_col = self.client._client.collections.get(self.collection_name)
                            if local_col and hasattr(local_col, "close"):
                                try:
                                    local_col.close()
                                except Exception:
                                    pass
                        self.client.delete_collection(self.collection_name)
                        exists = False
                else:
                    info = self.client.get_collection(self.collection_name)
                    current_size = None
                    if hasattr(info.config.params.vectors, 'size'):
                        current_size = info.config.params.vectors.size
                    elif isinstance(info.config.params.vectors, dict) and 'size' in info.config.params.vectors:
                        current_size = info.config.params.vectors['size']
                    
                    if current_size and current_size != self.vector_size:
                        # If writable, or if recreate was explicitly requested, or if collection is a test collection, recreate
                        if self.writable or should_recreate or "test" in self.collection_name:
                            print(f"Recreating collection '{self.collection_name}' due to vector size change ({current_size} -> {self.vector_size})...", file=sys.stderr)
                            if hasattr(self.client, "_client") and hasattr(self.client._client, "collections"):
                                local_col = self.client._client.collections.get(self.collection_name)
                                if local_col and hasattr(local_col, "close"):
                                    try:
                                        local_col.close()
                                    except Exception:
                                        pass
                            self.client.delete_collection(self.collection_name)
                            self.recreated_due_to_dim_change = True
                            exists = False
                        else:
                            print(
                                f"Notice: Vector dimension mismatch ({current_size} on disk vs {self.vector_size} in current embedder). "
                                "Preserving existing collection; skipping dense search for this query.",
                                file=sys.stderr
                            )
                            self.dimension_mismatch = True
                            return
        except Exception as e:
            print(f"Warning checking collection existence: {e}", file=sys.stderr)
            exists = False
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

        # Create payload indexes for fast filtered searches in server mode (local mode indexes automatically)
        if not self.is_in_memory_fallback:
            for field_name, schema_type in [
                ("repo_id", PayloadSchemaType.KEYWORD),
                ("file_paths", PayloadSchemaType.KEYWORD),
                ("source_type", PayloadSchemaType.KEYWORD),
                ("is_reverted", PayloadSchemaType.BOOL),
                ("timestamp_unix", PayloadSchemaType.INTEGER),
            ]:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema_type
                    )
                except Exception:
                    pass

    def upsert_chunks(self, chunks: List[dict], embeddings: List[List[float]], batch_size: int = 100):
        """Upserts chunks and their embeddings to Qdrant in batches of batch_size."""
        if not embeddings:
            return
        actual_size = len(embeddings[0])
        if actual_size != self.vector_size:
            print(f"Mismatch between vector_store.vector_size ({self.vector_size}) and actual embedding size ({actual_size}). Adjusting collection...", file=sys.stderr)
            self.vector_size = actual_size
            self.init_collection()

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            raw_id = chunk["id"]
            point_id = raw_id
            if isinstance(raw_id, str):
                try:
                    uuid.UUID(raw_id)
                except Exception:
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))
            c_repo = chunk.get("repo_id")
            c_clean = c_repo.strip().lower() if c_repo else ""
            c_base = c_clean.split("/")[-1] if c_clean else ""
            repo_tags = list(dict.fromkeys([x for x in [c_repo, c_clean, c_base] if x]))
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "id": raw_id,
                    "chunk_id": raw_id,
                    "repo_id": c_repo,
                    "repo_base": c_base,
                    "repo_aliases": repo_tags,
                    "source_type": chunk["source_type"],
                    "source_id": chunk["source_id"],
                    "text": chunk["text"],
                    "timestamp": chunk["timestamp"].isoformat() if hasattr(chunk["timestamp"], "isoformat") else chunk["timestamp"],
                    "timestamp_unix": int(chunk["timestamp"].timestamp()) if hasattr(chunk["timestamp"], "timestamp") else 0,
                    "file_paths": chunk.get("file_paths", []),
                    "symbols_modified": chunk.get("symbols_modified", []),
                    "related_ids": chunk.get("related_ids", []),
                    "is_reverted": chunk.get("is_reverted", False),
                }
            ))
        
        if points:
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )

    def search_chunks(
        self, 
        query_vector: List[float], 
        limit: int = 10,
        file_path: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        source_types: Optional[List[str]] = None,
        is_reverted: Optional[bool] = None,
        repo_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Searches collection with vector similarity and payload metadata filters."""
        must_filters = []

        if repo_id:
            clean = repo_id.strip().lower()
            base = clean.split("/")[-1]
            candidate_repos = {clean, base, repo_id}
            try:
                from archaeologist.storage.db import get_session_context
                from archaeologist.storage.models import Commit
                from archaeologist.utils.security import escape_like
                from sqlmodel import select
                with get_session_context() as session:
                    escaped_base = escape_like(base)
                    stmt = select(Commit.repo_id).where(
                        (Commit.repo_id == clean) |
                        (Commit.repo_id.like(f"%/{escaped_base}", escape="\\")) |
                        (Commit.repo_id == base)
                    ).limit(10)
                    for r_id in session.exec(stmt).all():
                        if r_id:
                            candidate_repos.add(r_id)
            except Exception:
                pass
            c_list = list(candidate_repos)
            repo_conditions = [
                FieldCondition(key="repo_id", match=MatchAny(any=c_list) if len(c_list) > 1 else MatchValue(value=c_list[0])),
                FieldCondition(key="repo_base", match=MatchValue(value=base)),
                FieldCondition(key="repo_aliases", match=MatchValue(value=base))
            ]
            must_filters.append(Filter(should=repo_conditions))
        
        if file_path:
            norm_fp = file_path.replace("\\", "/")
            candidate_paths = [norm_fp]
            fp_base = os.path.basename(norm_fp)
            
            try:
                from archaeologist.storage.db import get_session_context
                from archaeologist.storage.models import Chunk
                from archaeologist.utils.security import escape_like
                from sqlmodel import select
                with get_session_context() as session:
                    escaped_base = escape_like(fp_base)
                    stmt = select(Chunk.file_paths).where(Chunk.file_paths.like(f"%{escaped_base}%", escape="\\"))
                    if repo_id:
                        stmt = stmt.where(Chunk.repo_id == repo_id)
                    results = session.exec(stmt.limit(50)).all()
                    db_paths = set()
                    for fp_list in results:
                        if isinstance(fp_list, list):
                            for p in fp_list:
                                if isinstance(p, str):
                                    if p == norm_fp or p.endswith("/" + norm_fp) or p.endswith("/" + fp_base) or p == fp_base:
                                        db_paths.add(p)
                    if db_paths:
                        candidate_paths = list(db_paths)
            except Exception:
                pass

            if len(candidate_paths) > 1:
                must_filters.append(FieldCondition(key="file_paths", match=MatchAny(any=candidate_paths)))
            else:
                must_filters.append(FieldCondition(key="file_paths", match=MatchValue(value=candidate_paths[0])))

            
        if is_reverted is not None:
            must_filters.append(FieldCondition(key="is_reverted", match=MatchValue(value=is_reverted)))

            
        if source_types:
            must_filters.append(FieldCondition(key="source_type", match=MatchAny(any=source_types)))
            
        def _safe_parse_datetime(dt_val: Any) -> Optional[datetime]:
            if not dt_val:
                return None
            if isinstance(dt_val, datetime):
                return dt_val
            if not isinstance(dt_val, str):
                return None
            clean_str = dt_val.strip()
            if clean_str.endswith("Z"):
                clean_str = clean_str[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(clean_str)
            except Exception:
                import re
                m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", clean_str)
                if m:
                    try:
                        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                    except Exception:
                        pass
            return None

        if date_from or date_to:
            range_filter = {}
            if date_from:
                parsed_from = _safe_parse_datetime(date_from)
                if parsed_from:
                    range_filter["gte"] = int(parsed_from.timestamp())
            if date_to:
                parsed_to = _safe_parse_datetime(date_to)
                if parsed_to:
                    range_filter["lte"] = int(parsed_to.timestamp())
            if range_filter:
                must_filters.append(FieldCondition(key="timestamp_unix", range=Range(**range_filter)))
            
        query_filter = Filter(must=must_filters) if must_filters else None
        
        try:
            if hasattr(self.client, "query_points"):
                res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=limit
                )
                results = res.points
            elif hasattr(self.client, "search"):
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit
                )
            else:
                results = []
        except Exception as e:
            print(f"Error executing Qdrant search: {e}", file=sys.stderr)
            results = []
        
        hits = []
        for hit in results:
            payload = dict(hit.payload) if isinstance(hit.payload, dict) else {}
            orig_id = payload.get("id") or payload.get("chunk_id") or hit.id
            if "id" not in payload:
                payload["id"] = orig_id
            hits.append({
                "id": orig_id,
                "score": hit.score,
                "payload": payload
            })
        return hits

    def close(self, force: bool = False):
        """Closes the underlying client connection. Shared local clients remain pooled unless force=True."""
        try:
            if not self.is_in_memory_fallback or force:
                raw_path = self.storage_path or get_default_qdrant_path()
                local_path = os.path.abspath(raw_path)
                with _CLIENT_LOCK:
                    _LOCAL_CLIENTS.pop(local_path, None)
                if hasattr(self.client, "close"):
                    self.client.close()
        except Exception:
            pass

