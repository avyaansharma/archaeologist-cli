import os
import sys
import json
import urllib.request
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

class Embedder:
    def __init__(self, gemini_key: Optional[str] = None, voyage_key: Optional[str] = None, openai_key: Optional[str] = None):
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
        # Default to local fastembed for ultra-fast, zero-quota, offline embedding.
        # Cloud providers can be explicitly selected via EMBEDDER_PROVIDER or USE_GEMINI_EMBEDDINGS.
        embed_provider_env = os.getenv("EMBEDDER_PROVIDER", "").lower()
        use_gemini = os.getenv("USE_GEMINI_EMBEDDINGS") == "1" or embed_provider_env == "gemini"
        use_voyage = embed_provider_env == "voyage" or bool(voyage_key)
        use_openai = embed_provider_env == "openai" or bool(openai_key)

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
            self.dimension = 384
            print("Embedder initialized with FastEmbed local model (BAAI/bge-small-en-v1.5).", file=sys.stderr)

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

    def embed_texts(self, texts: List[str], return_success_flags: bool = False):
        """Embeds a list of texts and returns a list of embeddings (list of floats).
        If return_success_flags is True, returns a tuple (embeddings, success_flags).
        """
        if not texts:
            return ([], []) if return_success_flags else []

        if self.model == "fastembed":
            res = self._embed_fastembed(texts)
            success = [True] * len(res)
        elif self.model == "models/gemini-embedding-001":
            res, success = self._embed_gemini(texts)
        elif self.model == "voyage-code-2":
            res, success = self._embed_voyage(texts)
        elif self.model == "text-embedding-3-small":
            res, success = self._embed_openai(texts)
        else:
            res = self._embed_mock(texts)
            success = [False] * len(res)

        if return_success_flags:
            return res, success
        return res

    def _embed_fastembed(self, texts: List[str]) -> List[List[float]]:
        if not hasattr(self, "_fastembed_model"):
            from fastembed import TextEmbedding
            self._fastembed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        embeddings = list(self._fastembed_model.embed(texts))
        return [e.tolist() for e in embeddings]

    def _embed_gemini(self, texts: List[str]):
        from google import genai
        import time

        embeddings = []
        success_flags = []
        total = len(texts)
        current_key_idx = 0
        exhausted_keys = set()

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
                key_idx = (current_key_idx + key_attempt) % len(self.gemini_keys)
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
                        current_key_idx = key_idx
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



    def _embed_voyage(self, texts: List[str]):
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

    def _embed_openai(self, texts: List[str]):
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

    def _embed_mock(self, texts: List[str]) -> List[List[float]]:
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


