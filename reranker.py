"""Cross-encoder reranker for lancedb-pro.

Supports SiliconFlow, Jina, Voyage, Pinecone, DashScope rerank endpoints.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class Reranker:
    """Async cross-encoder reranker client.

    Provider formats supported:
    - siliconflow / jina:  POST /rerank  body={query, documents: str[]}
                            resp: {results: [{index, relevance_score}]}
    - voyage:              same body, resp: {data: [{index, relevance_score}]}
    - pinecone:            body={query, text: str[]}  resp: {data: [{index, score}]}
    - dashscope:           body={query, documents: str[]}
                            resp: {data: {results: [{index, relevance_score}]}}
    - tei:                 body={texts: str[], query: str}  resp: [{index, score}]
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://api.siliconflow.cn/v1/rerank",
        model: str = "BAAI/bge-reranker-v2-m3",
        provider: str = "siliconflow",
        timeout_ms: int = 5000,
    ):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.provider = provider.lower()
        self.timeout = timeout_ms / 1000.0
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={"Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Rerank documents against query.

        Returns list of (original_index, relevance_score) sorted by score desc.
        """
        if not documents:
            return []
        if top_n is None:
            top_n = len(documents)

        client = await self._get_client()
        url = self.endpoint

        # Build request per provider format
        if self.provider in ("siliconflow", "jina", "voyage"):
            body = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": False,
            }
        elif self.provider == "pinecone":
            body = {
                "model": self.model,
                "query": query,
                "text": documents,
                "top_n": top_n,
            }
        elif self.provider == "dashscope":
            body = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": False,
            }
        elif self.provider == "tei":
            body = {
                "model": self.model,
                "query": query,
                "texts": documents,
                "top_n": top_n,
            }
        else:
            body = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }

        resp = await client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()

        # Parse response per provider
        if self.provider == "siliconflow" or self.provider == "jina":
            results = data.get("results", data.get("data", []))
            out = [(r["index"], r["relevance_score"]) for r in results]
        elif self.provider == "voyage":
            results = data.get("data", [])
            out = [(r["index"], r["relevance_score"]) for r in results]
        elif self.provider == "pinecone":
            results = data.get("data", [])
            out = [(r["index"], r["score"]) for r in results]
        elif self.provider == "dashscope":
            inner = data.get("data", {}).get("results",
                                            data.get("results", []))
            out = [(r["index"], r["relevance_score"]) for r in inner]
        elif self.provider == "tei":
            # TEI returns a flat list ordered by score desc
            out = [(r["index"], r["score"]) for r in data]
        else:
            results = data.get("results", data.get("data", []))
            out = [(r["index"], r.get("relevance_score", r.get("score", 0)))
                   for r in results]

        return sorted(out, key=lambda x: x[1], reverse=True)
