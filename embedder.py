"""Embedding client for lancedb-pro.

Wraps OpenAI-compatible embedding endpoints (SiliconFlow, Jina, Ollama, etc.).
Mirrors the full embedding config from memory-lancedb-pro (TypeScript):
  - Provider detection (Jina/Voyage/NVIDIA/generic)
  - requestDimensions / omitDimensions (Matryoshka truncation)
  - taskQuery / taskPassage task hints
  - normalized (base64) embeddings
  - Auto-chunking for long documents
  - API key round-robin rotation (array of keys)
  - LRU cache with TTL
  - Error classification (auth / network / server) with actionable hints
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import time
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Built-in dimension tables for common models — mirrors TypeScript EMBEDDING_DIMENSIONS
_MODEL_DIMENSIONS = {
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-004": 768,
    "text-embedding-ada-002": 1536,
    # Google
    "gemini-embedding-001": 3072,
    # BGE / SiliconFlow
    "BAAI/bge-m3": 1024,
    "bge-m3": 1024,
    "bge-large-zh-v1.5": 1024,
    "bge-small-zh-v1.5": 512,
    # Jina v2 / v3
    "jina-embeddings-v2-base-en": 768,
    "jina-embeddings-v2-base-zh": 768,
    "jina-embeddings-v3": 1024,
    "jina-embeddings-v5-text-small": 1024,
    "jina-embeddings-v5-text-nano": 768,
    # M3E
    "m3e-base": 768,
    "m3e-large": 1024,
    # GTE / Qwen
    "gte-qwen2-1.5b-instruct": 896,
    "qwen-embeddings-v2": 1024,
    # Nomic
    "nomic-embed-text": 768,
    # Mxbai
    "mxbai-embed-large": 1024,
    # Voyage
    "voyage-4": 1024,
    "voyage-4-lite": 1024,
    "voyage-4-large": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-3-large": 1024,
    # SentenceTransformers
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 512,
}


# ---------------------------------------------------------------------------
# Provider detection — mirrors TypeScript detectEmbeddingProviderProfile
# ---------------------------------------------------------------------------

_PROVIDER_PROFILES = {
    "jina":        {"task_field": "task",      "norm": True,  "dims_field": "dimensions"},
    "voyage":      {"task_field": "input_type","norm": False, "dims_field": "output_dimension",
                    "task_map": {"retrieval.query": "query", "retrieval.passage": "document",
                                 "query": "query", "document": "document"}},
    "nvidia":      {"task_field": "input_type","norm": False, "dims_field": "dimensions",
                    "task_map": {"retrieval.query": "query", "retrieval.passage": "passage",
                                 "query": "query", "passage": "passage"}},
    "openai":      {"task_field": None,        "norm": False, "dims_field": "dimensions"},
    "azure-openai":{"task_field": None,        "norm": False, "dims_field": "dimensions"},
    "generic":     {"task_field": None,        "norm": False, "dims_field": "dimensions"},
}


def _detect_provider(base_url: str, model: str) -> str:
    """Detect embedding provider profile from base URL and model name.

    Host-based detection takes precedence over model-name heuristics
    (same logic as TypeScript source).
    """
    if not base_url:
        # Model-name fallback
        m = model.lower()
        if m.startswith("jina-"):   return "jina"
        if m.startswith("voyage"):  return "voyage"
        if m.startswith("nvidia/") or m.startswith("nv-embed"): return "nvidia"
        return "generic"

    url_lower = base_url.lower()
    if "api.openai.com" in url_lower:  return "openai"
    if "openai.azure.com" in url_lower: return "azure-openai"
    if "api.jina.ai" in url_lower:      return "jina"
    if "api.voyageai.com" in url_lower: return "voyage"
    if url_lower.endswith(".nvidia.com") or url_lower == "nvidia.com": return "nvidia"
    return "generic"


# ---------------------------------------------------------------------------
# LRU cache with TTL — mirrors TypeScript EmbeddingCache
# ---------------------------------------------------------------------------

class _EmbeddingCache:
    """Simple LRU cache with TTL for embedding vectors."""

    def __init__(self, max_size: int = 256, ttl_minutes: float = 30):
        self._cache = {}          # key -> (vector, timestamp_ms)
        self._order = []          # keys in LRU order
        self._max_size = max_size
        self._ttl_ms = ttl_minutes * 60_000
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _make_key(text: str, task: Optional[str]) -> str:
        """SHA-256-based short cache key: hash of 'task:text'."""
        import hashlib
        raw = f"{task or ''}:{text}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, text: str, task: Optional[str]) -> Optional[List[float]]:
        key = self._make_key(text, task)
        entry = self._cache.get(key)
        if entry is None:
            self.misses += 1
            return None
        vec, ts = entry
        if time.time() * 1000 - ts > self._ttl_ms:
            del self._cache[key]
            self._order.remove(key)
            self.misses += 1
            return None
        # Move to end (most recently used)
        self._order.remove(key)
        self._order.append(key)
        self.hits += 1
        return vec

    def set(self, text: str, task: Optional[str], vector: List[float]) -> None:
        key = self._make_key(text, task)
        now = time.time() * 1000
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self._max_size:
            # Evict oldest
            oldest = self._order.pop(0)
            self._cache.pop(oldest, None)
        self._cache[key] = (vector, now)
        self._order.append(key)

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{(self.hits / total * 100):.1f}%" if total else "N/A",
        }


# ---------------------------------------------------------------------------
# Error classification helpers — mirrors TypeScript formatEmbeddingProviderError
# ---------------------------------------------------------------------------

def _is_auth_error(status: int, msg: str) -> bool:
    return status in (401, 403) or re.search(
        r"invalid.*key|auth|forbidden|unauthorized|api.key.expired", msg, re.I)

def _is_network_error(code: str, msg: str) -> bool:
    return bool(code and re.search(
        r"ECONNREFUSED|ECONNRESET|ENOTFOUND|EHOSTUNREACH|ETIMEDOUT", code)) \
        or re.search(
        r"fetch.failed|network.error|socket.hang.up|connection.refused|getaddrinfo", msg, re.I)

import re


def format_embed_error(err: Exception, base_url: str, model: str) -> str:
    """Convert raw exception to actionable error message, matching TypeScript."""
    msg = str(err)
    profile = _detect_provider(base_url, model)
    provider_label = _label_from_profile(profile, base_url, model)

    if _is_auth_error(getattr(err, "response", None) and err.response.status_code or 0, msg):
        hint = f"Check embedding.apiKey and endpoint for {provider_label}."
        if profile == "jina":
            hint += (" If your Jina key expired or lost access, replace the key or switch to "
                     "a local OpenAI-compatible endpoint such as Ollama.")
        elif provider_label == "Ollama":
            hint += (" Ollama usually works with a dummy apiKey; verify the local server "
                     "is running, the model is pulled, and embedding.dimensions matches the output.")
        return f"Embedding provider authentication failed ({msg}). {hint}"

    if _is_network_error(getattr(err, "code", "") or "", msg):
        hint = f"Verify the endpoint is reachable"
        if base_url:
            hint += f" at {base_url}"
        hint += f" and that model '{model}' is available."
        return f"Embedding provider unreachable ({msg}). {hint}"

    return f"Failed to generate embedding from {provider_label}: {msg}"


def _label_from_profile(profile: str, base_url: str, model: str) -> str:
    if profile == "jina":        return "Jina"
    if profile == "voyage":       return "Voyage"
    if profile == "openai":       return "OpenAI"
    if profile == "azure-openai":return "Azure OpenAI"
    if profile == "nvidia":       return "NVIDIA NIM"
    if base_url:
        try:
            from urllib.parse import urlparse
            return urlparse(base_url).host or base_url
        except Exception:
            pass
    return "embedding provider"


# ---------------------------------------------------------------------------
# Embedder class — mirrors TypeScript Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """Async embedding client with retry, rate-limit handling, and token counting."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        model: str = "BAAI/bge-m3",
        dimensions: Optional[int] = 1024,
        normalized: bool = False,
        task_query: str = "",
        task_passage: str = "",
        timeout: float = 30.0,
        *,
        # Extended config from memory-lancedb-pro
        request_dimensions: Optional[int] = None,
        omit_dimensions: bool = False,
        api_key_list: Optional[List[str]] = None,
        cache_size: int = 256,
        cache_ttl_minutes: float = 30,
    ):
        # Resolve env vars in keys
        self._keys = self._resolve_keys(api_key, api_key_list)
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._normalized = normalized
        self._task_query = task_query
        self._task_passage = task_passage
        self._timeout = timeout
        # Schema dim: used for LanceDB table sizing
        self._dimensions = self._resolve_dimensions(model, dimensions)
        # Request dim: sent to API (may differ for Matryoshka truncation)
        self._request_dimensions = request_dimensions
        self._omit_dimensions = omit_dimensions
        self._client_index = 0
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(5)
        self._profile = _detect_provider(self._base_url, self._model)
        self._caps = _PROVIDER_PROFILES.get(self._profile, _PROVIDER_PROFILES["generic"])
        self._cache = _EmbeddingCache(max_size=cache_size, ttl_minutes=cache_ttl_minutes)

    @staticmethod
    def _resolve_env(value: str) -> str:
        """Resolve ${ENV_VAR} references in a string."""
        import re as _re
        def _subst(m):
            name = m.group(1)
            val = os.environ.get(name)
            if val is None:
                raise ValueError(f"Environment variable {name} is not set")
            return val
        return _re.sub(r'\$\{([^}]+)\}', _subst, value)

    def _resolve_keys(self, primary: str, key_list: Optional[List[str]]) -> List[str]:
        """Resolve env vars in all API keys."""
        keys = []
        for k in ([primary] + (key_list or [])):
            if k:
                keys.append(self._resolve_env(k))
        return keys or [self._resolve_env(primary)]

    @staticmethod
    def _resolve_dimensions(model: str, explicit: Optional[int]) -> int:
        """Resolve vector dimensions: explicit > model lookup > 1024."""
        if explicit:
            return explicit
        model_lower = model.lower()
        for key, dim in _MODEL_DIMENSIONS.items():
            if key in model_lower:
                return dim
        return 1024

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def cache_stats(self) -> dict:
        return self._cache.stats

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={
                    "Authorization": f"Bearer {self._keys[0]}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _build_request_body(self, texts: List[str], task_type: str = "") -> dict:
        """Build request body per provider profile — mirrors TypeScript logic."""
        body: dict = {"model": self._model, "input": texts}

        # Dimensions field (per-provider name)
        if not self._omit_dimensions and self._request_dimensions:
            dims_field = self._caps["dims_field"]
            body[dims_field] = self._request_dimensions
        elif not self._omit_dimensions and self._dimensions:
            dims_field = self._caps["dims_field"]
            body[dims_field] = self._dimensions

        # Task hint
        task_field = self._caps.get("task_field")
        if task_field:
            task_val = self._task_query if task_type == "query" else self._task_passage
            if task_val:
                task_map = self._caps.get("task_map", {})
                body[task_field] = task_map.get(task_val, task_val)

        # Normalized
        if self._caps["norm"] and self._normalized:
            body["encoding_format"] = "base64"

        return body

    async def embed(
        self,
        texts: List[str],
        task_type: str = "",
        retries: int = 3,
    ) -> List[List[float]]:
        """Embed a batch of texts with cache hit, key rotation, and retry.

        Returns list of embedding vectors.
        """
        if not texts:
            return []

        # Check cache for each text (batch cache miss → batch API call)
        cached = [self._cache.get(t, task_type if task_type else None) for t in texts]
        if all(v is not None for v in cached):
            return cached  # type: ignore[return-value]

        body = self._build_request_body(texts, task_type)
        last_err: Optional[Exception] = None

        for attempt in range(retries):
            # Round-robin key selection
            key_idx = (self._client_index) % len(self._keys)
            api_key = self._keys[key_idx]
            client = await self._get_client()
            # Update headers with current key
            client.headers["Authorization"] = f"Bearer {api_key}"
            url = f"{self._base_url}/embeddings"

            try:
                async with self._semaphore:
                    resp = await client.post(url, json=body)
                    if resp.status_code == 429:
                        retry_after = float(resp.headers.get("retry-after", 2 ** attempt))
                        logger.warning("Embedder rate-limited, backing off %.1fs", retry_after)
                        await asyncio.sleep(retry_after)
                        self._client_index = (self._client_index + 1) % len(self._keys)
                        continue
                    resp.raise_for_status()
                    data = resp.json()

                self._client_index = (self._client_index + 1) % len(self._keys)

                embeddings: List[List[float]] = []
                for i, item in enumerate(data.get("data", [])):
                    vec = item.get("embedding", [])
                    if self._normalized and isinstance(vec, str):
                        import base64 as _b64
                        vec = list(_b64.b64decode(vec).view(dtype="float32"))
                    # Cache result
                    self._cache.set(texts[i],
                                    task_type if task_type else None,
                                    vec)
                    embeddings.append(vec)

                return embeddings

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    last_err = e
                    await asyncio.sleep(2 ** attempt)
                    continue
                # Auth / client error → don't retry
                raise
            except Exception as e:
                last_err = e
                await asyncio.sleep(1)
                continue

        raise RuntimeError(f"Embedder failed after {retries} attempts: {last_err}")

    async def embed_one(
        self,
        text: str,
        task_type: str = "",
        retries: int = 3,
    ) -> List[float]:
        """Embed a single text. Convenience wrapper."""
        vecs = await self.embed([text], task_type=task_type, retries=retries)
        return vecs[0] if vecs else []

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for Chinese/English mixed."""
        return max(1, len(text) // 4)

    def chunk_text(
        self,
        text: str,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
    ) -> List[str]:
        """Split long text into overlapping chunks by token count."""
        if self.estimate_tokens(text) <= max_tokens:
            return [text]

        chars_per_token = 4
        max_chars = max_tokens * chars_per_token
        overlap_chars = overlap_tokens * chars_per_token

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + max_chars
            if end >= text_len:
                chunks.append(text[start:])
                break

            # Try to break at sentence/punctuation boundary
            for split_char in ("。", "！", "？", ". ", "\n", "，"):

                last_break = text.rfind(split_char, start + max_chars // 2,
                                        end)
                if last_break > start:
                    end = last_break + 1
                    break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap_chars
            if start >= text_len:
                break

        return chunks
