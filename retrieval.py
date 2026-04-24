"""Hybrid retrieval engine for lancedb-pro.

Combines vector search + BM25 with RRF fusion, then cross-encoder reranking.
Applies recency boost, time decay, importance weighting, and length normalization.

All methods are synchronous. Async embedding/reranking is handled internally
via _run_async helper.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import time
from typing import Any, Dict, List, Optional

from .decay import DecayEngine
from .reranker import Reranker
from .store import LanceDBStore

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Hybrid retrieval with all scoring stages from memory-lancedb-pro.

    All public methods are synchronous — async HTTP calls (embedding, rerank)
    are wrapped internally.
    """

    def __init__(
        self,
        store: LanceDBStore,
        embedder,  # Embedder instance
        reranker: Optional[Reranker],
        decay_engine: DecayEngine,
        mode: str = "hybrid",
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        min_score: float = 0.3,
        hard_min_score: float = 0.35,
        candidate_pool_size: int = 20,
        recency_half_life_days: float = 14.0,
        recency_weight: float = 0.1,
        filter_noise: bool = True,
        length_norm_anchor: int = 500,
        time_decay_half_life_days: float = 60.0,
        reinforcement_factor: float = 0.5,
        max_half_life_multiplier: float = 3.0,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.decay = decay_engine
        self.mode = mode
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.min_score = min_score
        self.hard_min_score = hard_min_score
        self.candidate_pool_size = candidate_pool_size
        self.recency_half_life_days = recency_half_life_days
        self.recency_weight = recency_weight
        self.filter_noise = filter_noise
        self.length_norm_anchor = length_norm_anchor
        self.time_decay_half_life_days = time_decay_half_life_days
        self.reinforcement_factor = reinforcement_factor
        self.max_half_life_multiplier = max_half_life_multiplier

    # ── Async helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _run_async(coro):
        """Run an async coroutine from synchronous code."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, coro).result(timeout=30)
        else:
            return asyncio.run(coro)

    # ── Main retrieval pipeline ───────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        scope_filter: Optional[List[str]] = None,
        category_filter: Optional[str] = None,
        include_vector: bool = False,
    ) -> List[Dict[str, Any]]:
        """Main retrieval pipeline: embed → search → scoring stages → rerank.

        Fully synchronous. Embedding/reranking HTTP calls are wrapped internally.
        """
        # 1. Embed query (async → sync)
        query_vec = self._run_async(
            self.embedder.embed_one(query, task_type="query"))

        # 2. Fetch candidates (all sync store calls)
        pool_size = max(self.candidate_pool_size, limit * 3)
        if self.mode == "hybrid":
            candidates = self.store.hybrid_search(
                query_vector=query_vec,
                query_text=query,
                vector_weight=self.vector_weight,
                bm25_weight=self.bm25_weight,
                limit=pool_size,
                scope_filter=scope_filter,
                category_filter=category_filter,
            )
        elif self.mode == "vector":
            candidates = self.store.vector_search(
                query_vector=query_vec,
                limit=pool_size,
                scope_filter=scope_filter,
                category_filter=category_filter,
            )
        else:
            candidates = self.store.bm25_search(
                query=query,
                limit=pool_size,
                scope_filter=scope_filter,
                category_filter=category_filter,
            )
        # 3. Filter by min_score
        candidates = [
            c for c in candidates
            if (c.get("score", 0) or c.get("_vector_score", 0) or 0) >= self.min_score
        ]
        # 4. Noise filter
        if self.filter_noise:
            from .extract import is_noise
            candidates = [c for c in candidates if not is_noise(c.get("text", ""))]
        # 5. Score enrichment: recency boost + time decay + importance + length norm
        now = time.time()
        scored = []
        for c in candidates:
            age_days = (now - c.get("created_at", now)) / 86400.0
            importance = c.get("importance", 0.5)
            confidence = c.get("confidence", 0.8)
            access_count = c.get("access_count", 0)

            # Time decay penalty
            decay_penalty = self.decay.time_decay_penalty(
                age_days, self.time_decay_half_life_days)
            base_score = c.get("score", 0) or c.get("_vector_score", 0) or 0

            # Recency boost (additive)
            recency_boost = self.decay.recency_boost(
                age_days, self.recency_half_life_days, self.recency_weight)

            # Lifecycle score
            lifecycle = self.decay.compute_lifecycle(
                importance, confidence, access_count, age_days)

            # Importance × confidence modulation
            importance_factor = min(1.0, importance * confidence * 1.5)

            # Length normalization
            text_len = len(c.get("text", ""))
            length_norm = self._length_norm(text_len, self.length_norm_anchor)

            # Final score
            final = (
                base_score
                * decay_penalty
                * (1.0 + recency_boost)
                * (0.5 + 0.5 * lifecycle)
                * importance_factor
                * length_norm
            )
            c["_final_score"] = final
            c["_age_days"] = age_days
            c["_lifecycle"] = lifecycle
            scored.append(c)

        # 6. Hard min score cutoff
        scored = [c for c in scored if c["_final_score"] >= self.hard_min_score]

        # 7. Rerank with cross-encoder if available
        if self.reranker and len(scored) > 1:
            scored = self._rerank(query, scored, limit)

        # Sort by final score
        scored.sort(key=lambda x: x.get("_final_score", 0), reverse=True)

        # 8. Record access for returned memories
        for mem in scored[:limit]:
            self.store.record_access(mem["id"])

        # Trim
        results = scored[:limit]

        # Optionally strip vectors to save space
        if not include_vector:
            for r in results:
                r.pop("vector", None)

        return results

    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Cross-encoder reranking (sync wrapper around async reranker)."""
        if not self.reranker:
            return candidates

        try:
            docs = [c.get("text", "") for c in candidates]
            reranked_scores = self._run_async(
                self.reranker.rerank(query, docs, top_n=len(candidates)))

            # Build reranked map: index → score
            rerank_map = {idx: score for idx, score in reranked_scores}

            # Normalize rerank scores to 0-1
            max_rerank = max(rerank_map.values()) if rerank_map else 1.0
            if max_rerank <= 0:
                max_rerank = 1.0

            # Get max final_score for normalization
            max_final = max((c.get("_final_score", 0) for c in candidates), default=1.0)
            if max_final <= 0:
                max_final = 1.0

            for i, c in enumerate(candidates):
                c["_rerank_score"] = rerank_map.get(i, 0) / max_rerank
                # Blend: 0.4 * rerank + 0.6 * normalized_final
                blended = (0.4 * c["_rerank_score"]
                           + 0.6 * (c.get("_final_score", 0) / max_final))
                c["_final_score"] = blended  # replace with blended

            candidates.sort(key=lambda x: x.get("_final_score", 0), reverse=True)
            return candidates

        except Exception as e:
            logger.warning("Reranking failed: %s", e)
            return candidates

    @staticmethod
    def _length_norm(char_len: int, anchor: int) -> float:
        """Length normalization: penalize very long entries.

        Formula: 1 / (1 + log2(charLen / anchor))
        anchor=500: len=250 → ~0.74, len=500 → ~0.5, len=2000 → ~0.29
        """
        if anchor <= 0 or char_len <= anchor:
            return 1.0
        return 1.0 / (1.0 + math.log2(char_len / anchor))

    def auto_recall(
        self,
        query: str,
        max_items: int = 3,
        max_chars: int = 600,
        per_item_max_chars: int = 180,
        min_length: int = 15,
    ) -> str:
        """Build auto-recall context string for injection into context.

        Returns formatted memory context string. Fully synchronous.
        """
        if len(query) < min_length:
            return ""

        # Truncate query to avoid huge embedding requests
        query = query[:2000]

        results = self.retrieve(query, limit=max_items)
        if not results:
            return ""

        lines = []
        total_chars = 0

        for mem in results:
            text = mem.get("text", "")
            category = mem.get("category", "other")

            # Per-item truncation
            if len(text) > per_item_max_chars:
                text = text[:per_item_max_chars] + "..."

            mem_text = f"[{category}] {text}"
            if total_chars + len(mem_text) > max_chars:
                # Truncate last one
                remaining = max_chars - total_chars
                if remaining > 50:
                    mem_text = mem_text[:remaining] + "..."
                    lines.append(mem_text)
                break

            lines.append(mem_text)
            total_chars += len(mem_text)

        if not lines:
            return ""

        return "## Memory Recall\n" + "\n".join(lines)
