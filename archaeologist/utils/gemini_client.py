import os
import sys
import json
import re
import time
import threading
import warnings
from typing import Optional, Dict, Any, List
from google import genai
from google.genai import types

warnings.filterwarnings("ignore", category=UserWarning, module="google.genai")
warnings.filterwarnings("ignore", message=".*automatic function calling.*")

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-lite"
FALLBACK_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]

def get_gemini_api_keys() -> List[str]:
    keys = []
    try:
        from archaeologist.storage.context import current_client_api_key_var
        ctx_key = current_client_api_key_var.get()
        if ctx_key and ctx_key.strip():
            keys.append(ctx_key.strip())
    except Exception:
        pass
    k1 = os.getenv("GEMINI_API_KEY")
    if k1 and k1 not in keys:
        keys.append(k1)
    k2 = os.getenv("GEMINI_API_KEY_SECONDARY")
    if k2 and k2 not in keys:
        keys.append(k2)
    k3 = os.getenv("GOOGLE_API_KEY")
    if k3 and k3 not in keys:
        keys.append(k3)
    return keys

def get_gemini_api_key() -> Optional[str]:
    keys = get_gemini_api_keys()
    return keys[0] if keys else None

class GeminiClientWrapper:
    _rate_limit_lock = threading.Lock()
    _last_request_time = 0.0
    _min_request_interval = 1.0     # Minimum 1.0 second delay between API calls
    _semaphore = threading.BoundedSemaphore(value=2)  # Max 2 concurrent API calls

    def __init__(self, api_key: Optional[str] = None):
        provided_key = api_key or get_gemini_api_key()
        self.api_keys = get_gemini_api_keys()
        if provided_key and provided_key not in self.api_keys:
            self.api_keys.insert(0, provided_key)
        if not self.api_keys:
            raise ValueError("GEMINI_API_KEY environment variable is required.")
            
        self.clients = [genai.Client(api_key=k) for k in self.api_keys]

    def _wait_for_rate_limit(self):
        """Enforces thread-safe rate throttling and interval pacing between requests."""
        with GeminiClientWrapper._rate_limit_lock:
            now = time.time()
            elapsed = now - GeminiClientWrapper._last_request_time
            if elapsed < GeminiClientWrapper._min_request_interval:
                time.sleep(GeminiClientWrapper._min_request_interval - elapsed)
            GeminiClientWrapper._last_request_time = time.time()

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_output_tokens: int = 2000
    ) -> str:
        """Generates text using Google Gemini API with API key rotation, model fallback, and rate throttling."""
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction if system_instruction else None
        )
        
        models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
        last_exception = None

        with GeminiClientWrapper._semaphore:
            self._wait_for_rate_limit()
            for client_idx, client in enumerate(self.clients):
                key_exhausted_or_invalid = False
                for m_name in models_to_try:
                    if key_exhausted_or_invalid:
                        break
                    try:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                client.models.generate_content,
                                model=m_name,
                                contents=prompt,
                                config=config
                            )
                            response = future.result(timeout=30.0)
                        return response.text.strip() if (response and response.text) else ""
                    except FutureTimeoutError:
                        print(f"Notice: Gemini call timed out after 30s on {m_name}. Trying next fallback model...", file=sys.stderr)
                        continue
                    except Exception as e:
                        last_exception = e
                        err_str = str(e)
                        if any(k in err_str for k in ["400", "401", "403", "API_KEY_INVALID", "PERMISSION_DENIED"]):
                            print(f"Notice: Gemini key at index {client_idx} failed auth ({err_str[:80]}). Rotating to next key...", file=sys.stderr)
                            key_exhausted_or_invalid = True
                            break
                        if any(k in err_str for k in ["404", "429", "500", "502", "503", "504", "NOT_FOUND", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "Quota exceeded", "not available"]):
                            print(f"Notice: Model {m_name} on key {client_idx} hit error/quota ({err_str[:60]}). Trying next fallback model...", file=sys.stderr)
                            continue
                        raise e

        raise last_exception or RuntimeError("Failed to generate content with available Gemini models/keys.")

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_output_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Generates JSON object using Google Gemini API with API key fallback and clean JSON parsing."""
        raw_text = self.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        
        # 1. Search for markdown code block anywhere in response
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
        if code_block:
            cleaned = code_block.group(1).strip()
        else:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
                cleaned = cleaned.strip()
            
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
            return {"raw_data": parsed}
        except json.JSONDecodeError as e:
            match = re.search(r"(\{[\s\S]*\})", cleaned)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            raise ValueError(f"Failed to parse JSON response from Gemini LLM: {raw_text}") from e

