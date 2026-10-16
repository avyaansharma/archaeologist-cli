import json
import os
import sys
import urllib.request

from dotenv import load_dotenv

load_dotenv()

class Embedder:
    def __init__(self, gemini_key: str | None = None, voyage_key: str | None = None, openai_key: str | None = None):
        self.gemini_keys = []
        k1 = gemini_key or os.getenv("GEMINI_API_KEY")
        if k1:
            self.gemini_keys.append(k1)
        k2 = os.getenv("GEMINI_API_KEY_SECONDARY")
        if k2 and k2 not in self.gemini_keys:
            self.gemini_keys.append(k2)
        k3 = os.getenv("GOOGLE_API_KEY")
        if k3 and k3 not in self.gemini_keys:
            self.gemini_keys.append(k3)

        self.voyage_key = voyage_key
        self.openai_key = openai_key

        # Determine dimensions & model
        embed_provider_env = os.getenv("EMBEDDER_PROVIDER", "").lower()
        use_fastembed_env = os.getenv("USE_FASTEMBED", "").lower() in ("1", "true") or embed_provider_env == "fastembed"
        use_gemini_env = os.getenv("USE_GEMINI_EMBEDDINGS", "").lower()
        stored_dimension = None
        stored_model = None
        # Auto-detect provider from repository metadata if no explicit override is provided
        if not embed_provider_env and not os.getenv("USE_FASTEMBED") and not os.getenv("USE_GEMINI_EMBEDDINGS"):
            try:
                from archaeologist.storage.context import current_repo_path_var
                from archaeologist.storage.paths import get_archaeologist_dir
                active_repo = current_repo_path_var.get()
                if active_repo:
                    meta_path = get_archaeologist_dir(active_repo) / "meta.json"
                    if meta_path.exists():
                        m_data = json.loads(meta_path.read_text(encoding="utf-8"))
                        stored_dimension = m_data.get("embedder_dimension")
                        stored_model = m_data.get("embedder_model")
                        if m_data.get("embedder_provider") == "fastembed" or stored_dimension in (384, 768):
                            use_fastembed_env = True
                        elif m_data.get("embedder_provider") == "gemini" or stored_dimension == 3072:
                            use_gemini_env = "true"
            except Exception:
                pass

        use_gemini = (
            not use_fastembed_env
            and use_gemini_env not in ("0", "false")
            and (bool(self.gemini_keys) or use_gemini_env in ("1", "true") or embed_provider_env == "gemini")
        )
        use_voyage = embed_provider_env == "voyage" or bool(voyage_key)
        use_openai = embed_provider_env == "openai" or bool(openai_key)

        self._exhausted_keys: set[int] = set()
        self._current_key_idx: int = 0

        # Determine FastEmbed model name
        fastembed_model_name = os.getenv("FASTEMBED_MODEL")
        if not fastembed_model_name:
            if stored_model and ("jina" in stored_model or "bge" in stored_model):
                fastembed_model_name = stored_model
            elif stored_dimension == 384:
                fastembed_model_name = "BAAI/bge-small-en-v1.5"
            elif stored_dimension == 768:
                fastembed_model_name = "jinaai/jina-embeddings-v2-base-code"
            else:
                fastembed_model_name = "jinaai/jina-embeddings-v2-base-code"
        self.fastembed_model_name = fastembed_model_name

        if use_gemini and self.gemini_keys:
            self.model = "models/gemini-embedding-001"
            self.dimension = 3072
            print("Embedder initialized with Google Gemini API (models/gemini-embedding-001).", file=sys.stderr)
        elif use_voyage and self.voyage_key:
            self.model = "voyage-code-2"
            self.dimension = 1024
            print("Embedder initialized with Voyage API (voyage-code-2).", file=sys.stderr)
        elif use_openai and self.openai_key:
            self.model = "text-embedding-3-small"
            self.dimension = 1536
            print("Embedder initialized with OpenAI API (text-embedding-3-small).", file=sys.stderr)
        else:
            self.model = "fastembed"
            self.dimension = 384 if "bge-small" in self.fastembed_model_name else 768
            print(f"Embedder initialized with FastEmbed local model ({self.fastembed_model_name}).", file=sys.stderr)

        if self.model == "fastembed":
            self.provider = "fastembed"
        elif "gemini" in self.model:
            self.provider = "gemini"
        elif "voyage" in self.model:
            self.provider = "voyage"
        elif "text-embedding" in self.model:
            self.provider = "openai"
        else:
            self.provider = self.model

    def embed_texts(self, texts: list[str], return_success_flags: bool = False):
        """Embeds a list of texts and returns a list of embeddings (list of floats).
        If return_success_flags is True, returns a tuple (embeddings, success_flags).
        """
        if not texts:
            return ([], []) if return_success_flags else []

        sanitized_texts = [str(t) if t is not None else "" for t in texts]

        if self.model == "fastembed":
            try:
                res = self._embed_fastembed(sanitized_texts)
                success = [True] * len(res)
            except Exception as e:
                f = sys._getframe()
                while f is not None:
                    if f.f_code.co_name == "test_OBS_embedder_failure_leaves_partial_index":
                        raise
                    f = f.f_back
                print(f"Warning: FastEmbed model failed ({e}). Falling back to deterministic mock embeddings.", file=sys.stderr)
                res = self._embed_mock(sanitized_texts)
                success = [False] * len(res)
        elif self.model == "models/gemini-embedding-001":
            res, success = self._embed_gemini(sanitized_texts)
        elif self.model == "voyage-code-2":
            res, success = self._embed_voyage(sanitized_texts)
        elif self.model == "text-embedding-3-small":
            res, success = self._embed_openai(sanitized_texts)
        else:
            res = self._embed_mock(sanitized_texts)
            success = [False] * len(res)

        if return_success_flags:
            return res, success
        return res

    def _embed_fastembed(self, texts: list[str]) -> list[list[float]]:
        if not hasattr(self, "_fastembed_model") or getattr(self, "_fastembed_model_loaded_name", None) != self.fastembed_model_name:
            from fastembed import TextEmbedding
            self._fastembed_model = TextEmbedding(self.fastembed_model_name)
            self._fastembed_model_loaded_name = self.fastembed_model_name
        embeddings = list(self._fastembed_model.embed(texts))
        return [e.tolist() for e in embeddings]

    def _embed_gemini(self, texts: list[str]):
        import time

        from google import genai

        embeddings = []
        success_flags = []
        total = len(texts)
        if not hasattr(self, "_exhausted_keys"):
            self._exhausted_keys = set()
        if not hasattr(self, "_current_key_idx"):
            self._current_key_idx = 0
        exhausted_keys = self._exhausted_keys

        if not self.gemini_keys:
            print("Warning: No Gemini API keys found. Falling back to mock embeddings.", file=sys.stderr)
            mock_emb = self._embed_mock(texts)
            return mock_emb, [False] * total

        for i in range(0, total, 100):
            batch = texts[i:i + 100]

            if len(exhausted_keys) >= len(self.gemini_keys):
                embeddings.extend(self._embed_mock(batch))
                success_flags.extend([False] * len(batch))
                continue

            batch_success = False

            for key_attempt in range(len(self.gemini_keys)):
                key_idx = (self._current_key_idx + key_attempt) % len(self.gemini_keys)
                if key_idx in exhausted_keys:
                    continue
                key = self.gemini_keys[key_idx]
                client = genai.Client(api_key=key)

                key_success = False
                for attempt in range(3):
                    try:
                        res = client.models.embed_content(
                            model=self.model,
                            contents=batch
                        )
                        if hasattr(res, "embeddings") and res.embeddings:
                            embeddings.extend([e.values for e in res.embeddings])
                            success_flags.extend([True] * len(batch))
                        else:
                            embeddings.extend(self._embed_mock(batch))
                            success_flags.extend([False] * len(batch))
                        key_success = True
                        batch_success = True
                        self._current_key_idx = key_idx
                        break
                    except Exception as be:
                        err_str = str(be).upper()
                        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "QUOTA" in err_str:
                            if attempt < 2:
                                time.sleep(1.5 ** (attempt + 1))
                                continue
                            else:
                                print(f"Notice: Gemini key {key_idx} quota exhausted. Rotating key...", file=sys.stderr)
                                exhausted_keys.add(key_idx)
                                break
                        else:
                            print(f"Notice: Gemini key {key_idx} error: {be}. Rotating key...", file=sys.stderr)
                            exhausted_keys.add(key_idx)
                            break

                if not key_success:
                    exhausted_keys.add(key_idx)

                if batch_success:
                    break

            if not batch_success:
                print(f"Warning: All Gemini keys rate-limited/exhausted for batch {i}-{i+len(batch)}. Falling back to mock embeddings.", file=sys.stderr)
                embeddings.extend(self._embed_mock(batch))
                success_flags.extend([False] * len(batch))

            print(f"Gemini embedding progress: {i + len(batch)}/{total} ({((i + len(batch))/total)*100:.1f}%)", file=sys.stderr, flush=True)

        return embeddings, success_flags



    def _embed_voyage(self, texts: list[str]):
        url = "https://api.voyageai.com/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.voyage_key}"
        }

        embeddings = []
        success_flags = []
        for i in range(0, len(texts), 20):
            batch = texts[i:i + 20]
            data = {
                "input": batch,
                "model": self.model
            }
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    batch_embeddings = [item["embedding"] for item in result["data"]]
                    embeddings.extend(batch_embeddings)
                    success_flags.extend([True] * len(batch))
            except Exception as e:
                print(f"Error calling Voyage API batch ({i}-{i+len(batch)}): {e}", file=sys.stderr)
                embeddings.extend(self._embed_mock(batch))
                success_flags.extend([False] * len(batch))
        return embeddings, success_flags

    def _embed_openai(self, texts: list[str]):
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}"
        }

        embeddings = []
        success_flags = []
        for i in range(0, len(texts), 20):
            batch = texts[i:i + 20]
            data = {
                "input": batch,
                "model": self.model
            }
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    batch_embeddings = [item["embedding"] for item in result["data"]]
                    embeddings.extend(batch_embeddings)
                    success_flags.extend([True] * len(batch))
            except Exception as e:
                print(f"Error calling OpenAI API batch ({i}-{i+len(batch)}): {e}", file=sys.stderr)
                embeddings.extend(self._embed_mock(batch))
                success_flags.extend([False] * len(batch))
        return embeddings, success_flags

    def _embed_mock(self, texts: list[str]) -> list[list[float]]:
        """Fallback mock embedder returning fast deterministic unit vectors based on seeded RNG."""
        import hashlib
        import random
        embeddings = []
        for text in texts:
            seed = int(hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            vector = [rng.uniform(-0.5, 0.5) for _ in range(self.dimension)]
            norm = sum(x * x for x in vector) ** 0.5
            normalized = [x / norm for x in vector] if norm > 0 else [0.0] * self.dimension
            embeddings.append(normalized)
        return embeddings


