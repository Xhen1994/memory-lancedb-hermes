"""LanceDB-Pro Memory Provider for Hermes Agent.

Python port of memory-lancedb-pro (CortexReach).
Hybrid vector + BM25 retrieval with cross-encoder reranking,
Weibull decay, auto-capture, and smart extraction.

Architecture:
  Embedder (SiliconFlow BAAI/bge-m3)
       → LanceDB Store (vector + FTS)
       → Retrieval Engine (hybrid RRF + scoring stages + rerank)
       → Smart Extractor (MiniMax LLM)
       → Auto-Capture (conversation → memories)

Config: $HERMES_HOME/lancedb_pro.json
Activate: memory.provider = "lancedb-pro" in config.yaml
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .config import LanceDBProConfig
from .decay import DecayEngine
from .embedder import Embedder
from .extract import (
    SmartExtractor, should_capture, detect_category, is_noise, clean_for_extraction,
)
from .reranker import Reranker
from .retrieval import RetrievalEngine
from .store import LanceDBStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

MEMORY_STORE_SCHEMA = {
    "name": "memory_store",
    "description": (
        "Store an explicit long-term memory fact. "
        "Use for preferences, decisions, user facts, project context you want to remember across sessions. "
        "Text should be concise and factual (20-200 chars). "
        "Categories: profile, preference, entity, decision, event, case, pattern, reflection, other"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The memory content to store."},
            "category": {
                "type": "string",
                "enum": ["profile", "preference", "entity", "decision",
                         "event", "case", "pattern", "reflection", "other"],
                "default": "other",
            },
            "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
            "scope": {"type": "string", "default": "global"},
        },
        "required": ["text"],
    },
}

MEMORY_SEARCH_SCHEMA = {
    "name": "memory_search",
    "description": (
        "Search long-term memories by meaning. Hybrid vector + BM25 search with cross-encoder reranking. "
        "Returns the most relevant memories ranked by score. "
        "Use for: recalling past discussions, preferences, decisions, facts about the user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (semantic or keyword)."},
            "category": {"type": "string", "description": "Filter by category."},
            "scope": {"type": "string", "description": "Filter by scope (default: global)."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
        },
        "required": ["query"],
    },
}

MEMORY_PROFILE_SCHEMA = {
    "name": "memory_profile",
    "description": (
        "Retrieve user profile memories — name, identity, preferences, background. "
        "Use at conversation start or when asked about the user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
        },
    },
}

MEMORY_STATS_SCHEMA = {
    "name": "memory_stats",
    "description": (
        "Show memory store statistics: total memories, category breakdown, "
        "storage health, and memory distribution by scope."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

MEMORY_FORGET_SCHEMA = {
    "name": "memory_forget",
    "description": (
        "Delete a memory by ID. Also accepts query to auto-find the best match to delete."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID to delete."},
            "query": {"type": "string", "description": "Query to find best-match memory to delete."},
        },
    },
}

MEMORY_LIST_SCHEMA = {
    "name": "memory_list",
    "description": "List stored memories, optionally filtered by scope or category.",
    "parameters": {
        "type": "object",
        "properties": {
            "scope": {"type": "string", "description": "Filter by scope."},
            "category": {"type": "string", "description": "Filter by category."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
        },
    },
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_json_config() -> Dict[str, Any]:
    """Load from $HERMES_HOME/lancedb_pro.json."""
    from hermes_constants import get_hermes_home
    hp = get_hermes_home()
    cfg_path = hp / "lancedb_pro.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_yaml_config() -> Dict[str, Any]:
    """Load from config.yaml plugins.lancedb-pro section."""
    from hermes_constants import get_hermes_home
    hp = get_hermes_home()
    cfg_path = hp / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path) as f:
                all_cfg = yaml.safe_load(f) or {}
            return all_cfg.get("plugins", {}).get("lancedb-pro", {}) or {}
        except Exception:
            pass
    return {}


def _make_provider_config() -> LanceDBProConfig:
    """Build config with correct precedence: json > yaml > defaults."""
    merged: Dict[str, Any] = {}
    merged.update(_load_yaml_config())
    merged.update(_load_json_config())
    return LanceDBProConfig(merged)

# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class LanceDBProProvider(MemoryProvider):
    """LanceDB-backed memory with full memory-lancedb-pro feature set."""

    name = "lancedb-pro"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = _make_provider_config() if config is None else LanceDBProConfig(config)
        self._store: Optional[LanceDBStore] = None
        self._embedder: Optional[Embedder] = None
        self._reranker: Optional[Reranker] = None
        self._decay: Optional[DecayEngine] = None
        self._extractor: Optional[SmartExtractor] = None
        self._retrieval: Optional[RetrievalEngine] = None
        self._initialized = False
        # Background prefetch state
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        # Turn counter for periodic maintenance
        self._turn_count = 0

    # -- MemoryProvider ABC ----------------------------------------------------

    def is_available(self) -> bool:
        try:
            import lancedb  # noqa: F401
            import httpx  # noqa: F401
            return True
        except ImportError:
            return False

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "dbPath", "description": "LanceDB data directory",
             "default": "~/.hermes/lancedb_pro_data"},
            {"key": "autoCapture", "description": "Auto-capture memories from conversation",
             "choices": ["true", "false"], "default": "true"},
            {"key": "autoRecall", "description": "Auto-inject memories into context",
             "choices": ["true", "false"], "default": "true"},
            {"key": "smartExtraction", "description": "Use LLM for memory extraction",
             "choices": ["true", "false"], "default": "true"},
            {"key": "enableManagementTools", "description": "Enable memory_stats and memory_list",
             "choices": ["true", "false"], "default": "true"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        cfg_path = Path(hermes_home) / "lancedb_pro.json"
        existing = {}
        if cfg_path.exists():
            try:
                existing = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(values)
        cfg_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    def initialize(self, session_id: str, **kwargs) -> None:
        if self._initialized:
            return
        cfg = self._config

        self._embedder = Embedder(
            api_key=cfg.emb_api_key,
            base_url=cfg.emb_base_url,
            model=cfg.emb_model,
            dimensions=cfg.emb_dimensions,
            normalized=cfg.emb_normalized,
            task_query=cfg.emb_task_query,
            task_passage=cfg.emb_task_passage,
        )

        self._reranker = Reranker(
            api_key=cfg.rerank_api_key,
            endpoint=cfg.rerank_endpoint,
            model=cfg.rerank_model,
            provider=cfg.rerank_provider,
            timeout_ms=cfg.rerank_timeout_ms,
        ) if cfg.rerank_enabled else None

        self._decay = DecayEngine(
            recency_half_life_days=cfg.decay_recency_half_life_days,
            recency_weight=cfg.decay_recency_weight,
            frequency_weight=cfg.decay_frequency_weight,
            intrinsic_weight=cfg.decay_intrinsic_weight,
            importance_modulation=cfg.decay_importance_modulation,
            beta_core=cfg.decay_beta_core,
            beta_working=cfg.decay_beta_working,
            beta_peripheral=cfg.decay_beta_peripheral,
        )

        self._extractor = SmartExtractor(
            llm_api_key=cfg.llm_api_key,
            llm_base_url=cfg.llm_base_url,
            llm_model=cfg.llm_model,
            timeout_ms=cfg.llm_timeout_ms,
        )

        self._store = LanceDBStore(
            db_path=cfg.db_path,
            embedder=self._embedder,
            vector_dim=cfg.emb_dimensions,
        )

        self._retrieval = RetrievalEngine(
            store=self._store,
            embedder=self._embedder,
            reranker=self._reranker,
            decay_engine=self._decay,
            mode=cfg.ret_mode,
            vector_weight=cfg.ret_vector_weight,
            bm25_weight=cfg.ret_bm25_weight,
            min_score=cfg.ret_min_score,
            hard_min_score=cfg.ret_hard_min_score,
            candidate_pool_size=cfg.ret_candidate_pool_size,
            recency_half_life_days=cfg.ret_recency_half_life_days,
            recency_weight=cfg.ret_recency_weight,
            filter_noise=cfg.ret_filter_noise,
            length_norm_anchor=cfg.ret_length_norm_anchor,
            time_decay_half_life_days=cfg.ret_time_decay_half_life_days,
            reinforcement_factor=cfg.ret_reinforcement_factor,
            max_half_life_multiplier=cfg.ret_max_half_life_multiplier,
        )

        # Initialize synchronous LanceDB store
        try:
            self._store.initialize()
        except Exception as e:
            logger.error("Failed to initialize LanceDB store: %s", e)

        self._initialized = True
        logger.info("LanceDB-Pro initialized (session=%s)", session_id)

    def system_prompt_block(self) -> str:
        if not self._initialized:
            return ""
        try:
            total = self._store.count()
            status = "active" if total > 0 else "active, empty"
            return (
                "# LanceDB Memory\n"
                f"Active ({status}). {total} memories stored with hybrid vector+BM25 retrieval.\n"
                "Use memory_store to save important facts.\n"
                "Use memory_search for semantic recall of past discussions."
            )
        except Exception:
            return "# LanceDB Memory\nActive."

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return prefetched recall context, or do synchronous recall."""
        if not self._initialized or not self._config.auto_recall:
            return ""

        # Check for background prefetch result first
        with self._prefetch_lock:
            if self._prefetch_result:
                result = self._prefetch_result
                self._prefetch_result = ""
                return result

        # Synchronous recall as fallback
        if len(query) < self._config.auto_recall_min_length:
            return ""
        try:
            result = self._retrieval.auto_recall(
                query,
                max_items=self._config.recall_max_items,
                max_chars=self._config.recall_max_chars,
                per_item_max_chars=self._config.auto_recall_per_item_max_chars,
                min_length=self._config.auto_recall_min_length,
            )
            return result or ""
        except Exception as e:
            logger.debug("Auto-recall failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background recall for the NEXT turn.

        Runs in a background thread so it doesn't block the current turn.
        """
        if not self._initialized or not self._config.auto_recall:
            return
        if len(query) < self._config.auto_recall_min_length:
            return

        def _bg_prefetch():
            try:
                result = self._retrieval.auto_recall(
                    query,
                    max_items=self._config.recall_max_items,
                    max_chars=self._config.recall_max_chars,
                    per_item_max_chars=self._config.auto_recall_per_item_max_chars,
                    min_length=self._config.auto_recall_min_length,
                )
                with self._prefetch_lock:
                    self._prefetch_result = result or ""
            except Exception as e:
                logger.debug("Background prefetch failed: %s", e)

        thread = threading.Thread(target=_bg_prefetch, daemon=True)
        thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = [MEMORY_STORE_SCHEMA, MEMORY_SEARCH_SCHEMA, MEMORY_PROFILE_SCHEMA]
        if self._config.enable_management_tools:
            schemas.extend([MEMORY_STATS_SCHEMA, MEMORY_LIST_SCHEMA])
        schemas.append(MEMORY_FORGET_SCHEMA)
        return schemas

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        handler_map = {
            "memory_store": self._handle_store,
            "memory_search": self._handle_search,
            "memory_profile": self._handle_profile,
            "memory_stats": self._handle_stats,
            "memory_forget": self._handle_forget,
            "memory_list": self._handle_list,
        }
        handler = handler_map.get(tool_name)
        if not handler:
            return tool_error(f"Unknown tool: {tool_name}")
        try:
            return handler(args)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e)
            return tool_error(str(e))

    # -- Per-turn hooks -------------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Track turn count for periodic maintenance."""
        self._turn_count = turn_number

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "") -> None:
        """Called after each turn — trigger-based real-time capture.

        Uses shouldCapture() keyword triggers for instant capture,
        without waiting for LLM extraction at session end.
        """
        if not self._initialized or not self._config.auto_capture:
            return
        if not self._store:
            return

        # Check if user message triggers auto-capture
        if user_content and should_capture(user_content):
            try:
                text = clean_for_extraction(user_content)
                if text and len(text) >= 10 and not is_noise(text):
                    # Truncate for storage
                    text = text[:300]
                    category = detect_category(text)
                    self._store.add(
                        text=text,
                        category=category,
                        importance=0.7,
                        confidence=0.8,
                        scope="global",
                    )
                    logger.debug("Trigger-captured memory: [%s] %s",
                                 category, text[:80])
            except Exception as e:
                logger.debug("Trigger capture failed: %s", e)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """End-of-session smart extraction using LLM."""
        if not self._initialized or not self._config.auto_capture:
            return
        if not messages or len(messages) < self._config.extract_min_messages:
            return
        try:
            # Run in background thread to not block shutdown
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            ex.submit(self._do_session_extraction, messages)
            # Wait briefly for extraction to complete, then proceed
            ex.shutdown(wait=True, cancel_futures=False)
        except Exception as e:
            logger.error("Auto-capture failed: %s", e)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract memories from messages about to be discarded by compression.

        Returns text to include in the compression summary.
        """
        if not self._initialized or not self._config.auto_capture:
            return ""
        if not messages:
            return ""

        try:
            # Do a quick extraction of the messages being compressed
            self._do_session_extraction(messages)
            return "Note: Key facts from compressed messages have been saved to long-term memory."
        except Exception as e:
            logger.debug("Pre-compress extraction failed: %s", e)
            return ""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to LanceDB."""
        if action != "add" or not self._initialized or not self._store:
            return
        try:
            self._store.add(
                text=content[:500],
                category="preference" if target == "user" else "other",
                importance=0.8,
                confidence=0.9,
            )
        except Exception as e:
            logger.debug("Memory write mirror failed: %s", e)

    def shutdown(self) -> None:
        try:
            if self._store:
                self._store.close()
            if self._embedder:
                import asyncio
                try:
                    asyncio.run(self._embedder.close())
                except Exception:
                    pass
            if self._reranker:
                import asyncio
                try:
                    asyncio.run(self._reranker.close())
                except Exception:
                    pass
            if self._extractor:
                import asyncio
                try:
                    asyncio.run(self._extractor.close())
                except Exception:
                    pass
        except Exception:
            pass
        self._initialized = False

    # -- Tool handlers (all synchronous) --------------------------------------

    def _handle_store(self, args: Dict[str, Any]) -> str:
        mid = self._store.add(
            text=args["text"],
            category=args.get("category", "other"),
            importance=float(args.get("importance", 0.5)),
            scope=args.get("scope", "global"),
            confidence=0.9,
        )
        if mid:
            return json.dumps({"memory_id": mid, "status": "stored"})
        return tool_error("Failed to store memory")

    def _handle_search(self, args: Dict[str, Any]) -> str:
        scope_filter = [args["scope"]] if args.get("scope") else None
        results = self._retrieval.retrieve(
            query=args["query"],
            limit=int(args.get("limit", 10)),
            scope_filter=scope_filter,
            category_filter=args.get("category"),
        )
        if results is None:
            return tool_error("Search failed")
        formatted = [
            {
                "id": r.get("id", ""),
                "text": r.get("text", ""),
                "category": r.get("category", "other"),
                "score": round(r.get("_final_score", r.get("score", 0)), 3),
                "importance": r.get("importance", 0),
            }
            for r in results
        ]
        return json.dumps({"results": formatted, "count": len(formatted)})

    def _handle_profile(self, args: Dict[str, Any]) -> str:
        results = self._retrieval.retrieve(
            query="user profile identity name background preferences habits",
            limit=int(args.get("limit", 10)),
            category_filter="profile",
        )
        if results is None:
            return tool_error("Profile fetch failed")
        formatted = [
            {"id": r.get("id", ""), "text": r.get("text", ""),
             "importance": r.get("importance", 0)}
            for r in results
        ]
        return json.dumps({"profile": formatted, "count": len(formatted)})

    def _handle_stats(self, args: Dict[str, Any]) -> str:
        if not self._initialized:
            return tool_error("Not initialized")
        total = self._store.count()
        category_counts: Dict[str, int] = {}
        scope_counts: Dict[str, int] = {}
        try:
            all_mems = self._store.list_memories(limit=1000) or []
            for m in all_mems:
                cat = m.get("category", "other")
                category_counts[cat] = category_counts.get(cat, 0) + 1
                scope = m.get("scope", "global")
                scope_counts[scope] = scope_counts.get(scope, 0) + 1
        except Exception:
            pass
        return json.dumps({
            "total_memories": total,
            "by_category": category_counts,
            "by_scope": scope_counts,
            "config": {
                "retrieval_mode": self._config.ret_mode,
                "vector_weight": self._config.ret_vector_weight,
                "bm25_weight": self._config.ret_bm25_weight,
                "auto_recall": self._config.auto_recall,
                "auto_capture": self._config.auto_capture,
                "smart_extraction": self._config.smart_extraction,
            },
        })

    def _handle_forget(self, args: Dict[str, Any]) -> str:
        memory_id = args.get("memory_id")
        query = args.get("query")
        if not memory_id and not query:
            return tool_error("Either memory_id or query is required")
        if memory_id:
            deleted = self._store.delete(memory_id)
            return json.dumps({"deleted": bool(deleted), "memory_id": memory_id})
        results = self._retrieval.retrieve(query=query, limit=1)
        if not results:
            return tool_error(f"No memory found matching: {query}")
        mid = results[0]["id"]
        deleted = self._store.delete(mid)
        return json.dumps({
            "deleted": bool(deleted),
            "memory_id": mid,
            "matched_text": results[0].get("text", "")[:100],
        })

    def _handle_list(self, args: Dict[str, Any]) -> str:
        memories = self._store.list_memories(
            scope=args.get("scope"),
            category=args.get("category"),
            limit=int(args.get("limit", 20)),
        )
        if memories is None:
            return tool_error("List failed")
        formatted = [
            {
                "id": m.get("id", ""),
                "text": m.get("text", ""),
                "category": m.get("category", "other"),
                "importance": m.get("importance", 0),
                "access_count": m.get("access_count", 0),
            }
            for m in memories
        ]
        return json.dumps({"memories": formatted, "count": len(formatted)})

    # -- Session extraction ---------------------------------------------------

    def _do_session_extraction(self, messages: List[Dict[str, Any]]) -> None:
        """Extract and store memories from conversation messages.

        Called by both on_session_end and on_pre_compress.
        """
        if not self._store or not self._extractor:
            return

        turns = []
        total_chars = 0
        max_chars = self._config.extract_max_chars
        for msg in messages:
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue
            if role == "assistant" and not self._config.capture_assistant:
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle structured content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = " ".join(text_parts)
            if not content or len(content) < 10:
                continue
            turns.append({"role": role, "content": content})
            total_chars += len(content)
            if total_chars > max_chars:
                break

        if len(turns) < self._config.extract_min_messages:
            return

        extracted = []
        if self._config.smart_extraction:
            try:
                extracted = self._extractor.extract_memories_sync(
                    turns, max_chars=max_chars)
            except Exception as e:
                logger.warning("Smart extraction failed: %s", e)
                extracted = []

        stored = 0
        for item in extracted:
            text = str(item.get("text", ""))[:300]
            if not text:
                continue
            try:
                self._store.add(
                    text=text,
                    category=item.get("category", "other"),
                    importance=float(item.get("importance", 0.5)),
                    confidence=float(item.get("confidence", 0.7)),
                    scope="global",
                )
                stored += 1
            except Exception as e:
                logger.debug("Failed to store extracted memory: %s", e)
        if stored:
            logger.info("Auto-captured %d memories from session", stored)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Called by the plugin loader to register this provider."""
    provider = LanceDBProProvider()
    ctx.register_memory_provider(provider)
