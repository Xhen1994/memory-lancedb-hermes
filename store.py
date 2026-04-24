"""LanceDB storage layer for lancedb-pro.

Uses synchronous LanceDB (lancedb >= 0.4) with LanceModel for schema.
Fully synchronous — callers that need async wrap via run_in_executor.
Thread-safe via file locking for writes.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Memory categories — mirrors memory-lancedb-pro
MEMORY_CATEGORIES = [
    "profile", "preference", "entity", "decision", "event", "case", "pattern",
    "reflection", "other"
]
DEFAULT_SCOPE = "global"


class LanceDBStore:
    """Thread-safe LanceDB-backed memory store.

    Uses synchronous LanceDB connection with LanceModel schema.
    ALL public methods are synchronous.
    """

    def __init__(
        self,
        db_path: str,
        embedder,
        vector_dim: int = 1024,
    ):
        self.db_path = db_path
        self.embedder = embedder
        self.vector_dim = vector_dim
        self._db = None
        self._table = None
        self._lock_path = os.path.join(db_path, ".lock")
        self._initialized = False
        self._schema_cls = None

    def initialize(self) -> None:
        """Initialize LanceDB connection synchronously."""
        if self._initialized:
            return

        Path(self.db_path).mkdir(parents=True, exist_ok=True)

        import lancedb

        self._db = lancedb.connect(self.db_path)

        # Define LanceModel schema
        try:
            from lancedb.pydantic import LanceModel, Vector
        except ImportError:
            raise RuntimeError(
                "lancedb[pydantic] required. Run: pip install 'lancedb[pydantic]'"
            )

        dim = self.vector_dim

        class MemorySchema(LanceModel):
            id: str
            text: str
            vector: Vector(dim)  # pyarrow FixedSizeList<float>
            category: str = "other"
            scope: str = DEFAULT_SCOPE
            importance: float = 0.5
            confidence: float = 0.8
            access_count: int = 0
            last_access: float = 0.0
            created_at: float = 0.0
            updated_at: float = 0.0
            metadata: str = "{}"

        self._schema_cls = MemorySchema

        table_name = "memories"
        raw = self._db.list_tables()
        # LanceDB 0.30+ returns a ListTablesResponse object; extract table names
        if hasattr(raw, 'tables'):
            existing = raw.tables
        elif isinstance(raw, dict):
            existing = raw.get("tables", [])
        else:
            existing = list(raw)
        if table_name in existing:
            self._table = self._db.open_table(table_name)
        else:
            self._table = self._db.create_table(
                table_name,
                schema=MemorySchema,
                exist_ok=True,
            )
            # Create FTS index for text (simple tokenizer supports Chinese)
            try:
                self._table.create_fts_index(
                    "text",
                    replace=True,
                    base_tokenizer="simple",
                    stem=False,
                    remove_stop_words=False,
                    ngram_min_length=1,
                    ngram_max_length=2,
                )
                logger.info("FTS index created on 'text' (simple tokenizer)")
            except Exception as e:
                logger.warning("FTS index creation skipped: %s", e)

        self._initialized = True
        logger.info("LanceDB store initialized at %s", self.db_path)

    def close(self) -> None:
        """Close the LanceDB connection."""
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None
            self._table = None
            self._initialized = False

    # ── File lock ─────────────────────────────────────────────────────────────

    def _acquire_lock(self, timeout: float = 10.0) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return True
            except FileExistsError:
                time.sleep(0.05)
        return False

    def _release_lock(self) -> None:
        try:
            os.remove(self._lock_path)
        except FileNotFoundError:
            pass

    # ── CRUD (all synchronous) ────────────────────────────────────────────────

    def add(
        self,
        text: str,
        vector: Optional[List[float]] = None,
        category: str = "other",
        scope: str = DEFAULT_SCOPE,
        importance: float = 0.5,
        confidence: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a memory. Embedding is done synchronously via _embed_sync.

        Returns the memory ID.
        """
        if not self._initialized:
            self.initialize()
        if not self._acquire_lock():
            raise RuntimeError("Could not acquire write lock")
        try:
            now = time.time()
            mid = str(uuid.uuid4())
            vec = vector if vector is not None else self._embed_sync(text)
            record = self._schema_cls(
                id=mid, text=text, vector=vec,
                category=category, scope=scope,
                importance=float(importance), confidence=float(confidence),
                access_count=0, last_access=float(now),
                created_at=float(now), updated_at=float(now),
                metadata=json.dumps(metadata or {}, ensure_ascii=False),
            )
            self._table.add([record])
            return mid
        finally:
            self._release_lock()

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a memory by ID."""
        if not self._initialized:
            self.initialize()
        try:
            result = list(self._table.query()
                          .where(f'id = "{memory_id}"').limit(1).to_list())
            return self._row_to_dict(result[0]) if result else None
        except Exception:
            return None

    def update(self, memory_id: str, **kwargs) -> bool:
        """Update fields of a memory by ID."""
        if not self._initialized:
            self.initialize()
        if not self._acquire_lock():
            raise RuntimeError("Could not acquire write lock")
        try:
            existing = self.get(memory_id)
            if not existing:
                return False
            updates = {"updated_at": time.time()}
            for k, v in kwargs.items():
                if v is not None:
                    updates[k] = v
            updates_str = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v)
                           for k, v in updates.items()}
            self._table.update().where(f'id = "{memory_id}"').set(updates_str).execute()
            return True
        finally:
            self._release_lock()

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if not self._initialized:
            self.initialize()
        try:
            self._table.delete(f'id = "{memory_id}"')
            return True
        except Exception:
            return False

    def record_access(self, memory_id: str) -> None:
        """Increment access count and update last_access timestamp.

        Uses read-modify-write instead of SQL expressions for compatibility.
        """
        if not self._initialized:
            self.initialize()
        try:
            existing = self.get(memory_id)
            if not existing:
                return
            new_count = existing.get("access_count", 0) + 1
            now = time.time()
            self._table.update().where(f'id = "{memory_id}"').set({
                "access_count": new_count,
                "last_access": now,
            }).execute()
        except Exception as e:
            logger.debug("record_access failed for %s: %s", memory_id, e)

    def count(self) -> int:
        """Total memory count."""
        if not self._initialized:
            self.initialize()
        try:
            return self._table.count_rows()
        except Exception:
            return 0

    def list_memories(
        self,
        scope: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List memories with optional filters."""
        if not self._initialized:
            self.initialize()
        q = self._table.query().limit(limit + offset)
        if scope:
            q = q.where(f'scope = "{scope}"')
        if category:
            q = q.where(f'category = "{category}"')
        results = list(q.to_list())
        return [self._row_to_dict(r) for r in results[offset:offset + limit]]

    # ── Search ─────────────────────────────────────────────────────────────────

    def vector_search(
        self,
        query_vector: List[float],
        limit: int = 20,
        scope_filter: Optional[List[str]] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Vector similarity search."""
        if not self._initialized:
            self.initialize()
        try:
            q = self._table.search(query_vector, vector_column_name="vector")
            if scope_filter:
                scope_cond = " OR ".join(f'scope = "{s}"' for s in scope_filter)
                q = q.where(f"({scope_cond})")
            if category_filter:
                q = q.where(f'category = "{category_filter}"')

            results = q.limit(limit * 3).to_list()
            scored = []
            for row in results:
                vec = row.get("vector", [])
                score = self._cosine_sim(query_vector, vec) if vec else 0.0
                d = self._row_to_dict(row)
                d["_vector_score"] = score
                scored.append(d)

            scored.sort(key=lambda x: x["_vector_score"], reverse=True)
            return scored[:limit]
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return []

    def bm25_search(
        self,
        query: str,
        limit: int = 20,
        scope_filter: Optional[List[str]] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Pure Python BM25 full-text search — works for any language incl. Chinese."""
        import re, math

        if not self._initialized:
            self.initialize()

        def tokenize(text: str) -> List[str]:
            # Chinese char-level + English/Alphanumeric tokenization
            chars = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]{2,}', text.lower())
            return chars

        def _bm25_score(
            query_tokens: List[str],
            doc_tokens: List[str],
            N: int,
            avgdl: float,
            doc_lens: List[int],
            idf_map: Dict[str, float],
            k1: float = 1.5,
            b: float = 0.75,
        ) -> float:
            dl = len(doc_tokens)
            score = 0.0
            for term in query_tokens:
                tf = doc_tokens.count(term)
                if tf == 0:
                    continue
                idf = idf_map.get(term, 0)
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
            return score

        try:
            all_rows = self._table.to_pandas().to_dict("records")
        except Exception as e:
            logger.error("BM25 failed to read table: %s", e)
            return []

        # Filter scope/category first
        filtered = []
        for row in all_rows:
            if scope_filter and row.get("scope") not in scope_filter:
                continue
            if category_filter and row.get("category") != category_filter:
                continue
            filtered.append(row)

        if not filtered:
            return []

        texts = [str(r.get("text", "")) for r in filtered]
        doc_tokens_list = [tokenize(t) for t in texts]
        doc_lens = [len(dt) for dt in doc_tokens_list]
        avgdl = sum(doc_lens) / len(doc_lens)
        N = len(texts)

        # IDF per query term
        query_tokens = tokenize(query)
        idf_map = {}
        for term in set(query_tokens):
            n = sum(1 for dt in doc_tokens_list if term in dt)
            idf_map[term] = math.log((N - n + 0.5) / (n + 0.5) + 1)

        # Score all docs
        scored = []
        for i, row in enumerate(filtered):
            score = _bm25_score(query_tokens, doc_tokens_list[i], N, avgdl, doc_lens, idf_map)
            d = self._row_to_dict(row)
            d["_bm25_score"] = score
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:limit]]

    def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str = "",
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        limit: int = 20,
        scope_filter: Optional[List[str]] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: RRF fusion of vector + BM25 results."""
        vec_results = self.vector_search(
            query_vector, limit=limit, scope_filter=scope_filter,
            category_filter=category_filter)
        bm25_results = self.bm25_search(
            query_text, limit=limit, scope_filter=scope_filter,
            category_filter=category_filter)

        fused: Dict[str, Dict[str, Any]] = {}
        for i, r in enumerate(vec_results):
            rid = r["id"]
            fused[rid] = {**r, "_vector_rank": i + 1}
        for i, r in enumerate(bm25_results):
            rid = r["id"]
            if rid in fused:
                fused[rid]["_bm25_rank"] = i + 1
                fused[rid]["_bm25_score"] = r.get("_bm25_score", 0)
            else:
                fused[rid] = {**r, "_bm25_rank": i + 1}

        k = 60  # RRF constant
        for d in fused.values():
            vr = d.get("_vector_rank", limit + 1)
            br = d.get("_bm25_rank", limit + 1)
            d["_rrf_score"] = vector_weight / (k + vr) + bm25_weight / (k + br)

        max_vec = max((d["_vector_score"] for d in fused.values()
                       if "_vector_score" in d), default=1.0) or 1.0
        max_bm = max((d["_bm25_score"] for d in fused.values()
                      if "_bm25_score" in d), default=1.0) or 1.0
        for d in fused.values():
            d["vector_score"] = d.get("_vector_score", 0) / max_vec
            d["bm25_score"] = d.get("_bm25_score", 0) / max_bm
            d["score"] = d["_rrf_score"]

        results = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        return list(results)[:limit]

    # ── Embedding helper ─────────────────────────────────────────────────────

    def _embed_sync(self, text: str, task_type: str = "") -> List[float]:
        """Synchronously get embedding by running the async embedder."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an event loop — run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run,
                                 self.embedder.embed_one(text, task_type=task_type)
                                 ).result(timeout=30)
        else:
            return asyncio.run(self.embedder.embed_one(text, task_type=task_type))

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _row_to_dict(row: Any) -> Dict[str, Any]:
        """Convert LanceDB row (LanceModel/dict) to plain dict."""
        if hasattr(row, "model_dump"):
            d = row.model_dump()
        elif hasattr(row, "__dict__"):
            d = dict(row.__dict__)
        else:
            d = dict(row)
        if "metadata" in d and isinstance(d["metadata"], str):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except Exception:
                pass
        if "vector" in d:
            vec = d["vector"]
            if hasattr(vec, "tolist"):
                d["vector"] = vec.tolist()
            elif hasattr(vec, "__iter__") and not isinstance(vec, (list, tuple, str)):
                try:
                    d["vector"] = list(vec)
                except Exception:
                    pass
        return d
